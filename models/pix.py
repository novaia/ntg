'''
WIP.
'''

import flax.linen as nn
import optax
import jax
import jax.numpy as jnp
import math

embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2
channels = 1

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

        upsample_shape = (x.shape[0] * 2, x.shape[1] * 2, x.shape[2])
        x = jax.image.resize(x, upsample_shape, method='bilinear')

        for _ in range(self.block_depth):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(self.width)(x, train)
        return x

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
        axis = 2
    )
    return embeddings

class DDIM(nn.Module):
    widths: list[int]
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
    
key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
x = jax.random.uniform(key1, (28, 28, 1))
noise_variances = jax.random.uniform(key2, (1, 1, 1))

model = DDIM(widths, block_depth)
params = model.init(key2, (x, noise_variances))