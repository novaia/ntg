'''
This is the model and training code for
Pix: a denoising diffusion implicit model (DDIM).
'''
import os
import pathlib
import sys
project_root = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
if project_root not in sys.path: sys.path.append(project_root)
os.chdir(project_root)

import argparse
import math
import functools
from datetime import datetime
from typing import Any
import flax.linen as nn
from flax.training import train_state
import optax # Optimizers.
#import orbax.checkpoint 
import jax
import jax.numpy as jnp
#import fid
import fid
from inference import reverse_diffusion
from keras.preprocessing.image import ImageDataGenerator

starting_epoch = 0 # 0 if training from scratch.
data_path = '../../heightmaps/uncorrupted_split_heightmaps_second_pass'
model_save_path = '../data/models/diffusion_models/'
model_name = 'pix'
image_save_path = '../data/images/'

# Sampling.
min_signal_rate = 0.02
max_signal_rate = 0.95

# Architecture.
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# Optimization.
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

# Input.
batch_size = 8
image_width = 256
image_height = 256
channels = 1

def preprocessing_function(image):
    image = image.astype(float) / 255
    return image

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = jnp.exp(
        jnp.linspace(
            jnp.log(embedding_min_frequency),
            jnp.log(embedding_max_frequency),
            embedding_dims // 2
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = jnp.concatenate(
        [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
        axis = -1
    )
    return embeddings

class ResidualBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x, train: bool):
        input_width = x.shape[-1]
        if input_width == self.width:
            residual = x
        else:
            residual = nn.Conv(self.width, kernel_size=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Conv(self.width, kernel_size=(3, 3))(x)
        x = nn.activation.swish(x)
        x = nn.Conv(self.width, kernel_size=(3, 3))(x)
        x = x + residual
        return x

class DownBlock(nn.Module):
    width: int
    block_depth: int

    @nn.compact
    def __call__(self, x, train: bool):
        x, skips = x

        for _ in range(self.block_depth):
            x = ResidualBlock(self.width)(x, train)
            skips.append(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

class UpBlock(nn.Module):
    width: int
    block_depth: int

    @nn.compact
    def __call__(self, x, train: bool):
        x, skips = x

        upsample_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
        x = jax.image.resize(x, upsample_shape, method='bilinear')

        for _ in range(self.block_depth):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(self.width)(x, train)
        return x

class DDIM(nn.Module):
    widths: list
    block_depth: int

    @nn.compact
    def __call__(self, x, train: bool):
        x, noise_variances = x

        e = sinusoidal_embedding(noise_variances)
        e = jax.image.resize(e, shape=x.shape, method='nearest')
        
        x = nn.Conv(self.widths[0], kernel_size=(1, 1))(x)
        x = jnp.concatenate([x, e], axis=-1)

        skips = []
        for width in self.widths[:-1]:
            x = DownBlock(width, self.block_depth)([x, skips], train)

        for _ in range(self.block_depth):
            x = ResidualBlock(self.widths[-1])(x, train)

        for width in reversed(self.widths[:-1]):
            x = UpBlock(width, self.block_depth)([x, skips], train)

        x = nn.Conv(channels, kernel_size=(1, 1), kernel_init=nn.initializers.zeros_init())(x)
        return x
    
def diffusion_schedule(diffusion_times):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates    

# Training.
class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(module, rng, learning_rate):
    x = (jnp.ones([1, image_width, image_height, 1]), jnp.ones([1, 1, 1, 1]))
    variables = module.init(rng, x, True)
    params = variables['params']
    batch_stats = variables['batch_stats']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx, batch_stats=batch_stats)

@jax.jit
def train_step(state, images, parent_key):
    noise_key, diffusion_time_key = jax.random.split(parent_key, 2)
    batch_size = len(images)
    
    def loss_fn(params):
        noises = jax.random.normal(noise_key, (batch_size, image_width, image_height, channels))
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch_size, 1, 1, 1))
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats}, 
             [noisy_images, noise_rates**2],
             train=True,
             mutable=['batch_stats']
        )

        # TODO: switch to MAE
        loss = jnp.mean((pred_noises - noises)**2)
        return loss, updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return loss, state

def fid_benchmark(apply_fn, params, batch_stats, stats_path, batch_size, num_samples):
    batch_size = 20
    num_samples = 5000
    num_batches = num_samples // batch_size

    def get_sample_batch(apply_fn, params, batch_stats, batch_size):
        samples = reverse_diffusion(
            apply_fn=apply_fn, 
            params=params,
            batch_stats=batch_stats, 
            num_images=batch_size, 
            diffusion_steps=10, 
            image_height=256, 
            image_width=256, 
            channels=1, 
            diffusion_schedule_fn=diffusion_schedule,
        )
        # FID requires 3 channels.
        return samples.repeat(3, axis=-1)
    get_batch_fn = functools.partial(
        get_sample_batch, apply_fn, params, batch_stats, batch_size
    )

    params, apply_fn = fid.get_inception_model()
    mu1, sigma1 = fid.compute_statistics_with_mmap(
        params = params, 
        apply_fn = apply_fn, 
        num_batches = num_batches,
        batch_size = batch_size, 
        get_batch_fn = get_batch_fn,
        filename = 'data/temp/mmap_file',
        dtype = 'float32', 
        num_activations = num_samples
    )
    mu2, sigma2 = fid.load_statistics(stats_path)
    fid_value = fid.compute_frechet_distance(mu1, mu2, sigma1, sigma2)
    print('FID:', fid_value)
    return fid_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'If true, use FID to benchmark model at the end of every epoch.'
    parser.add_argument('--use_fid', type=bool, default=True, help=help_text)
    help_text = 'Path to file containing precomputed FID stats for training dataset.'
    parser.add_argument('--fid_stats_path', type=str, help=help_text)
    help_text = 'Batch size for generating FID samples.'
    parser.add_argument('--fid_batch_size', type=int, help=help_text)
    help_text = 'Number of samples to generate for FID benchmark.'
    parser.add_argument('--num_fid_samples', type=int, help=help_text)
    args = parser.parse_args()

    if args.use_fid:
        fid.check_for_correct_setup(args.fid_stats_path)

    print('GPU:', jax.devices('gpu'))

    init_rng = jax.random.PRNGKey(0)
    model = DDIM(widths, block_depth)
    state = create_train_state(model, init_rng, learning_rate)
    del init_rng

    fid_benchmark(
        apply_fn = state.apply_fn, 
        params = state.params, 
        batch_stats = state.batch_stats, 
        stats_path = args.fid_stats_path,
        batch_size = args.fid_batch_size,
        num_samples = args.num_fid_samples
    )
    exit(0)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    if starting_epoch != 0:
        checkpoint_path = model_save_path + model_name + '_epoch' + str(starting_epoch - 1)
        state = checkpointer.restore(checkpoint_path, state)

    idg = ImageDataGenerator(preprocessing_function = preprocessing_function)
    heightmap_iterator = idg.flow_from_directory(
        data_path, 
        target_size = (image_height, image_width), 
        batch_size = batch_size,
        color_mode = 'grayscale',
        classes = ['']
    )

    epochs = 3
    steps_per_epoch = len(heightmap_iterator)

    losses = []
    for epoch in range(epochs):
        epoch_start_time = datetime.now()

        losses_this_epoch = []
        for step in range(steps_per_epoch):
            images = jnp.asarray(heightmap_iterator.next()[0])
            
            if images.shape[0] != batch_size:
                continue
            
            train_step_key = jax.random.PRNGKey(epoch * steps_per_epoch + step)
            loss, state = train_step(state, images, train_step_key)
            losses_this_epoch.append(loss)
        losses.append(sum(losses_this_epoch) / len(losses_this_epoch))

        epoch_end_time = datetime.now()
        epoch_delta_time = epoch_end_time - epoch_start_time
        simple_epoch_end_time = str(epoch_end_time.hour) + ':' + str(epoch_end_time.minute)
        absolute_epoch = starting_epoch + epoch

        print(
            'Epoch', 
            absolute_epoch, 
            'completed at', 
            simple_epoch_end_time, 
            'in', 
            str(epoch_delta_time)
        )

        save_name = model_save_path + model_name + '_epoch' + str(absolute_epoch+1)
        checkpointer.save(save_name, state)
    print('losses', losses)