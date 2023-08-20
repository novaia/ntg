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
import shutil
import math
import functools
from datetime import datetime
from typing import Any
import flax.linen as nn
from flax.training import train_state
import optax # Optimizers.
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import fid
from inference import reverse_diffusion
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

start_epoch = 0 # 0 if training from scratch.
dataset_path = '../heightmaps/world-heightmaps-01/'
model_save_path = 'data/pix_checkpoints/'
model_name = 'pix'
image_save_path = 'data/pix_training_generations/'
log_path = 'data/logs/pix.csv'
in_docker_container = True

# Sampling.
min_signal_rate = 0.02
max_signal_rate = 0.95

# Architecture.
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [128, 128, 128, 128]
block_depth = 2

# Optimization.
learning_rate = 1e-4
epochs = 100

# Input.
batch_size = 4
image_width = 256
image_height = 256
channels = 1

# Benchmarking.
use_fid = True
fid_stats_path = 'data/dataset_info/world-heightmaps-01-stats.npz'
fid_batch_size = 20
num_fid_samples = 5000

def preprocessing_function(image):
    image = image.astype(float) / 255
    image = (image * 2) - 1
    return image

# Transforms images from [-1, 1] to [0, 1].
def bipolar_to_binary(images):
    return (images + 1) / 2

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

def create_train_state(module, rng, learning_rate, image_width, image_height):
    x = (jnp.ones([1, image_width, image_height, 1]), jnp.ones([1, 1, 1, 1]))
    variables = module.init(rng, x, True)
    params = variables['params']
    batch_stats = variables['batch_stats']
    tx = optax.adam(learning_rate)
    train_state = TrainState.create(
        apply_fn=module.apply, 
        params=params, 
        tx=tx, 
        batch_stats=batch_stats
    )
    return train_state

@jax.jit
def train_step(state, images, parent_key):
    noise_key, diffusion_time_key = jax.random.split(parent_key, 2)
    batch_size = len(images)
    
    def loss_fn(params):
        noises = jax.random.normal(
            noise_key, (batch_size, images.shape[1], images.shape[2], channels)
        )
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
    num_batches = num_samples // batch_size

    def get_sample_batch(apply_fn, params, batch_stats, batch_size, seed):
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
            seed=seed
        )
        samples = bipolar_to_binary(samples)
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
    return fid_value

def save_checkpoint(
    checkpointer, state, model_save_path, checkpoint_name, in_docker_container
):
    final_save_path = os.path.join(model_save_path, checkpoint_name)
    if in_docker_container:
        # To read why this is necessary, see: https://github.com/google/orbax/issues/446
        # Save to root owned checkpoints dir.
        temp_save_path = os.path.abspath(os.path.join('../checkpoints', checkpoint_name))
        checkpointer.save(os.path.abspath(temp_save_path), state)
        # Copy from root owned checkpoints dir, to checkpoints dir in mounted volume.
        shutil.copytree(temp_save_path, final_save_path)
    else:
        checkpointer.save(final_save_path, state)

def save_generations(
    state, num_images, diffusion_steps, image_width, image_height, epoch, save_path
):
    samples = reverse_diffusion(
        apply_fn=state.apply_fn, 
        params=state.params,
        batch_stats=state.batch_stats, 
        num_images=num_images, 
        diffusion_steps=diffusion_steps, 
        image_height=image_height, 
        image_width=image_width, 
        channels=1, 
        diffusion_schedule_fn=diffusion_schedule,
        seed=0
    )
    samples = bipolar_to_binary(samples)
    samples = jnp.clip(samples, 0.0, 1.0)
    samples = samples.repeat(3, axis=-1)
    for i, sample in enumerate(samples):
        plt.imsave(os.path.join(save_path, f'{epoch}_{i}.png'), sample, cmap='gray')

def get_checkpoint_name(model_name, epoch):
    return f'{model_name}_epoch{epoch}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'Epoch to start from when resuming training. Starts from scratch if 0.'
    parser.add_argument('--start_epoch', type=int, default=start_epoch, help=help_text)
    help_text = 'Path to training dataset.'
    parser.add_argument('--dataset_path', type=str, default=dataset_path, help=help_text)
    help_text = 'Path to directory where model checkpoints are saved.'
    parser.add_argument('--model_save_path', type=str, default=model_save_path, help=help_text)
    help_text = 'Name of the model. Used for naming checkpoints and training logs.'
    parser.add_argument('--model_name', type=str, default=model_name, help=help_text)
    help_text = 'Path to directory where images generated at the end of each epoch are saved.'
    parser.add_argument('--image_save_path', type=str, default=image_save_path, help=help_text)
    help_text = 'Path to log file.'
    parser.add_argument('--log_file', type=str, default=log_path, help=help_text)
    help_text = 'Batch size for training.'
    parser.add_argument('--batch_size', type=int, default=batch_size, help=help_text)
    help_text = 'Learning rate for Adam optimizer.'
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help=help_text)
    help_text = 'Number of epochs to train for.'
    parser.add_argument('--epochs', type=int, default=epochs, help=help_text)
    help_text = 'Width of image to load from dataset.'
    parser.add_argument('--image_width', type=int, default=image_width, help=help_text)
    help_text = 'Height of image to load from dataset.'
    parser.add_argument('--image_height', type=int, default=image_height, help=help_text)
    help_text = 'If true, use FID to benchmark model at the end of every epoch.'
    parser.add_argument('--use_fid', type=bool, default=use_fid, help=help_text)
    help_text = 'Path to file containing precomputed FID stats for training dataset.'
    parser.add_argument('--fid_stats_path', type=str, default=fid_stats_path, help=help_text)
    help_text = 'Batch size for generating FID samples.'
    parser.add_argument('--fid_batch_size', type=int, default=fid_batch_size, help=help_text)
    help_text = 'Number of samples to generate for FID benchmark.'
    parser.add_argument('--num_fid_samples', type=int, default=num_fid_samples, help=help_text)
    help_text = 'If true, the program will expect to be running inside a docker container.'
    parser.add_argument('--docker', type=bool, default=in_docker_container, help=help_text)
    args = parser.parse_args()
    print('GPU:', jax.devices('gpu'))

    if args.use_fid:
        fid.check_for_correct_setup(args.fid_stats_path)

    if not os.path.isfile(args.log_file):
        with open(args.log_file, 'w+') as f:
            f.write('epoch,loss,fid\n')

    init_rng = jax.random.PRNGKey(0)
    model = DDIM(widths, block_depth)
    state = create_train_state(
        model, init_rng, args.learning_rate, args.image_width, args.image_height
    )
    del init_rng
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f'Parameter Count: {param_count}')

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if args.start_epoch != 0:
        checkpoint_name = get_checkpoint_name(args.model_name, args.start_epoch)
        checkpoint_path = os.path.join(args.model_save_path, checkpoint_name)
        state = checkpointer.restore(os.path.abspath(checkpoint_path), item=state)
        print(f'Resuming training from epoch {args.start_epoch}')
    else:
        print('Starting training from scratch')

    idg = ImageDataGenerator(preprocessing_function = preprocessing_function)
    heightmap_iterator = idg.flow_from_directory(
        args.dataset_path, 
        target_size = (args.image_height, args.image_width), 
        batch_size = args.batch_size,
        color_mode = 'grayscale',
        classes = ['']
    )
    steps_per_epoch = len(heightmap_iterator)

    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        absolute_epoch = args.start_epoch + epoch + 1

        losses = []
        for step in range(steps_per_epoch):
            images = jnp.asarray(heightmap_iterator.next()[0])
            
            if images.shape[0] != batch_size:
                continue
            
            train_step_key = jax.random.PRNGKey(absolute_epoch * steps_per_epoch + step)
            loss, state = train_step(state, images, train_step_key)
            losses.append(loss)
        average_loss = sum(losses) / len(losses)

        epoch_end_time = datetime.now()
        epoch_delta_time = epoch_end_time - epoch_start_time
        simple_epoch_end_time = str(epoch_end_time.hour) + ':' + str(epoch_end_time.minute)

        print(
            'Epoch', 
            absolute_epoch, 
            'completed at', 
            simple_epoch_end_time, 
            'in', 
            str(epoch_delta_time)
        )
        print(f'Loss: {average_loss}')

        checkpoint_name = get_checkpoint_name(args.model_name, absolute_epoch)
        save_checkpoint(
            checkpointer = checkpointer, 
            state = state, 
            model_save_path = args.model_save_path, 
            checkpoint_name = checkpoint_name,
            in_docker_container = args.docker
        )

        save_generations(
            state = state, 
            num_images = 10, 
            diffusion_steps = 20, 
            image_width = args.image_width, 
            image_height = args.image_height, 
            epoch = absolute_epoch, 
            save_path = args.image_save_path
        )

        fid_value = fid_benchmark(
            apply_fn = state.apply_fn, 
            params = state.params, 
            batch_stats = state.batch_stats, 
            stats_path = args.fid_stats_path,
            batch_size = args.fid_batch_size,
            num_samples = args.num_fid_samples
        )
        print('FID:', fid_value)
        
        with open(args.log_file, 'a') as f:
            f.write(f'{absolute_epoch},{average_loss},{fid_value}\n')