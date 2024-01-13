import jax
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
from orbax import checkpoint as ocp
import optax

from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali import types as dali_types

from PIL import Image
import pandas as pd

from typing import Any, Callable, List
from functools import partial
from datetime import datetime
import json
import math
import os

from models.common import config_utils
from sampling.diffusion import implicit as sample_implicit

def save_samples(samples:jax.Array, step:int, save_dir:str):
    samples = ((samples + 1.0) / 2.0) * 255.0
    samples = jnp.clip(samples, 0.0, 255.0)
    samples = np.array(samples, dtype=np.uint8)
    for i in range(samples.shape[0]):
        image = Image.fromarray(samples[i].squeeze(axis=-1))
        image.save(os.path.join(save_dir, f'step{step}_image{i}.png'))

def get_data_iterator(
    dataset_path:str, image_height:int, image_width:int, batch_size:int, num_threads:int = 3
):
    @pipeline_def
    def my_pipeline_def(files, image_height, image_width, file_root):
        image_files = fn.readers.file(
            files=files, 
            file_root=file_root, 
            read_ahead=True, 
            shuffle_after_epoch=True, 
            device='cpu'
        )[0]
        images = fn.decoders.image(
            image_files, 
            device='mixed', 
            output_type=dali_types.GRAY, 
            preallocate_height_hint=image_height,
            preallocate_width_hint=image_width
        )
        flipped = fn.flip(
            images, 
            vertical=fn.random.coin_flip(device='cpu'), 
            horizontal=fn.random.coin_flip(device='cpu'), 
            device='gpu'
        )
        casted = fn.cast(flipped, dtype=dali_types.FLOAT)
        scale_constant = dali_types.Constant(127.5).float32()
        shift_constant = dali_types.Constant(1.0).float32()
        normalized = (casted / scale_constant) - shift_constant
        return normalized
    
    file_names = list(pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))['file_name'])
    steps_per_epoch = len(file_names) // batch_size
    data_pipeline = my_pipeline_def(
        files=file_names,
        file_root=dataset_path,
        image_height=image_height,
        image_width=image_width,
        batch_size=batch_size, 
        num_threads=num_threads, 
        device_id=0
    )
    data_iterator = DALIGenericIterator(
        pipelines=[data_pipeline], output_map=['x'], last_batch_policy=LastBatchPolicy.DROP
    )
    return data_iterator, steps_per_epoch

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_max_frequency:float
    embedding_min_frequency:float = 1.0
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(self.embedding_min_frequency),
                jnp.log(self.embedding_max_frequency),
                self.embedding_dim // 2,
                dtype=self.dtype
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
            axis=-1,
            dtype=self.dtype
        )
        return embeddings

class ResidualBlock(nn.Module):
    num_features: int
    num_groups: int
    kernel_size: int
    activation_fn: Callable
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        input_features = x.shape[-1]
        if input_features == self.num_features:
            residual = x
        else:
            residual = nn.Conv(
                self.num_features, kernel_size=(1, 1), 
                dtype=self.dtype, param_dtype=self.param_dtype
            )(x)
        x = nn.Conv(
            self.num_features, kernel_size=(self.kernel_size, self.kernel_size), 
            dtype=self.dtype, param_dtype=self.param_dtype
        )(x)
        x = nn.GroupNorm(self.num_groups, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = self.activation_fn(x)
        x = nn.Conv(
            self.num_features, kernel_size=(self.kernel_size, self.kernel_size), 
            dtype=self.dtype, param_dtype=self.param_dtype
        )(x)
        x = self.activation_fn(x)
        x = x + residual
        return x

class DownBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    kernel_size: int
    activation_fn: Callable
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, skips):
        for _ in range(self.block_depth):
            x = ResidualBlock(
                num_features=self.num_features, 
                num_groups=self.num_groups,
                kernel_size=self.kernel_size,
                activation_fn=self.activation_fn,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )(x)
            skips.append(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skips

class UpBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    kernel_size: int
    activation_fn: Callable
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, skips):
        upsample_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
        x = jax.image.resize(x, upsample_shape, method='bilinear')

        for _ in range(self.block_depth):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(
                num_features=self.num_features,
                num_groups=self.num_groups,
                kernel_size=self.kernel_size,
                activation_fn=self.activation_fn,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )(x)
        return x, skips

class Terra(nn.Module):
    embedding_dim: int
    embedding_max_frequency: float
    num_features: List[int]
    num_groups: List[int]
    kernel_size: int
    block_depth: int
    output_channels: int
    activation_fn: Callable
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, diffusion_time):
        time_emb = SinusoidalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_max_frequency=self.embedding_max_frequency,
            dtype=self.dtype
        )(diffusion_time)
        time_emb = jax.image.resize(time_emb, shape=(*x.shape[0:3], 1), method='nearest')
        x = jnp.concatenate([x, time_emb], axis=-1)

        skips = []
        block_params = list(zip(self.num_features, self.num_groups))[:-1]
        for features, groups in block_params:
            x, skips = DownBlock(
                num_features=features,
                num_groups=groups,
                block_depth=self.block_depth,
                kernel_size=self.kernel_size,
                activation_fn=self.activation_fn,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )(x, skips)
        for _ in range(self.block_depth):
            x = ResidualBlock(
                num_features=self.num_features[-1],
                num_groups=self.num_groups[-1],
                kernel_size=self.kernel_size,
                activation_fn=self.activation_fn,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )(x)
        for features, groups in list(reversed(block_params)):
            x, skips = UpBlock(
                num_features=features,
                num_groups=groups,
                block_depth=self.block_depth,
                kernel_size=self.kernel_size,
                activation_fn=self.activation_fn,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )(x, skips)

        x = nn.Conv(
            self.output_channels, kernel_size=(1, 1), 
            dtype=jnp.float32, param_dtype=jnp.float32
        )(x)
        return x

def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

@partial(jax.jit, static_argnames=['min_signal_rate', 'max_signal_rate', 'noise_clip'])
def train_step(state, images, min_signal_rate, max_signal_rate, noise_clip, key):
    noise_key, diffusion_time_key = jax.random.split(key, 2)
    noises = jax.random.normal(noise_key, images.shape, dtype=jnp.float32)
    noises = jnp.clip(noises, -noise_clip, noise_clip)
    diffusion_times = jax.random.uniform(diffusion_time_key, (images.shape[0], 1, 1, 1))
    noise_rates, signal_rates = diffusion_schedule(
        diffusion_times, min_signal_rate, max_signal_rate
    )
    noisy_images = signal_rates * images + noise_rates * noises

    def loss_fn(params):
        pred_noises = state.apply_fn({'params': params}, noisy_images, noise_rates**2)
        return jnp.mean((pred_noises - noises)**2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def main():
    gpu = jax.devices('gpu')[0]
    print(gpu)

    args = config_utils.parse_args(default_run_dir='data/terra_runs/0')

    checkpoint_save_dir = os.path.join(args.run_dir, 'checkpoints')
    fixed_seed_save_dir = os.path.join(args.run_dir, 'images/fixed')
    dynamic_seed_save_dir = os.path.join(args.run_dir, 'images/dynamic')
    if not os.path.exists(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)
    if not os.path.exists(fixed_seed_save_dir):
        os.makedirs(fixed_seed_save_dir)
    if not os.path.exists(dynamic_seed_save_dir):
        os.makedirs(dynamic_seed_save_dir)

    with open(args.config, 'r') as f:
        config = json.load(f)
    assert len(config['num_features']) == len(config['num_groups']), (
        'len(num_features) must equal len(num_groups).'
    )
    
    data_iterator, steps_per_epoch = get_data_iterator(
        dataset_path=args.dataset, 
        image_height=config['image_size'],
        image_width=config['image_size'],
        batch_size=config['batch_size']
    )
    print('Steps per epoch', steps_per_epoch)

    activation_fn = config_utils.load_activation_fn(config['activation_fn'])
    dtype = config_utils.load_dtype(config['dtype'])
    param_dtype = config_utils.load_dtype(config['param_dtype'])

    model = Terra(
        embedding_dim=config['embedding_dim'],
        embedding_max_frequency=config['embedding_max_frequency'],
        num_features=config['num_features'],
        num_groups=config['num_groups'],
        block_depth=config['block_depth'],
        kernel_size=config['kernel_size'],
        output_channels=config['output_channels'],
        activation_fn=activation_fn,
        dtype=dtype,
        param_dtype=param_dtype
    )
    x = jnp.ones(
        (
            config['batch_size'], 
            config['image_size'], 
            config['image_size'], 
            config['output_channels']
        ),
        dtype=dtype
    )
    diffusion_times = jnp.ones((config['batch_size'], 1, 1, 1), dtype=dtype)
    model_key = jax.random.PRNGKey(0)
    if args.tabulate:
        print(model.tabulate(model_key, x, diffusion_times))
        exit(0)
    params = model.init(model_key, x, diffusion_times)['params']
    
    epochs_to_steps = partial(lambda steps, epochs: int(steps * epochs), steps=steps_per_epoch)
    lr_schedule = optax.warmup_exponential_decay_schedule(
        init_value=config['lr_base'],
        peak_value=config['lr_max'],
        warmup_steps=epochs_to_steps(epochs=config['lr_warmup_epochs']),
        transition_steps=epochs_to_steps(epochs=config['lr_decay_epochs']),
        decay_rate=config['lr_decay_rate'],
        staircase=False,
        end_value=config['lr_min']
    )
    tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=config['adam_b1'],
        b2=config['adam_b2'],
        eps=config['adam_eps'],
        eps_root=config['adam_eps_root'],
        weight_decay=config['weight_decay']
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    config['param_count'] = param_count
    print('Param count', param_count)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if args.checkpoint is not None:
        state = checkpointer.restore(args.checkpoint, item=state)

    if args.wandb == 1:
        import wandb
        wandb.init(project='ntg-terra', config=config)

    min_signal_rate = config['min_signal_rate']
    max_signal_rate = config['max_signal_rate']
    noise_clip = config['noise_clip']
    sample_fn = partial(
        sample_implicit,
        num_images=16,
        diffusion_steps=20,
        diffusion_schedule=diffusion_schedule,
        image_width=config['image_size'],
        image_height=config['image_size'],
        channels=config['output_channels'],
        min_signal_rate=min_signal_rate,
        max_signal_rate=max_signal_rate,
        noise_clip=noise_clip,
    )

    for epoch in range(config['epochs']):
        epoch_start_time = datetime.now()
        for _ in range(steps_per_epoch):
            images = next(data_iterator)['x']
            step_key = jax.random.PRNGKey(state.step)
            loss, state = train_step(
                state, images, min_signal_rate, max_signal_rate, noise_clip, step_key
            )
            if args.wandb == 1: 
                if state.step % args.steps_between_wandb_logs == 0:
                    wandb.log({'loss': loss}, step=state.step)
            else:
                print(state.step, loss)
        epoch_end_time = datetime.now()
        print(
            f'Epoch {epoch} completed in {epoch_end_time-epoch_start_time} at {epoch_end_time}'
        )

        if args.save_checkpoints == 1:
            checkpointer.save(
                os.path.join('/project/', checkpoint_save_dir, f'step{state.step}'), 
                state, force=True
            )
        if (epoch+1) % args.epochs_between_previews != 0:
            continue

        fixed_seed_samples = sample_fn(state=state, seed=0)
        dynamic_seed_samples = sample_fn(state=state, seed=state.step)
        save_samples(fixed_seed_samples, state.step, fixed_seed_save_dir)
        save_samples(dynamic_seed_samples, state.step, dynamic_seed_save_dir)

if __name__ == '__main__':
    main()
