"""
Frechet Inception Distance.
Mostly taken from: https://github.com/matthias-wright/jax-fid/blob/main/jax_fid/inception.py
License: https://github.com/matthias-wright/jax-fid/blob/main/LICENSE
The code in this file was modified from the original.
"""

import flax.linen as nn
import jax.numpy as jnp
import jax
import pickle
import functools
from typing import Callable, Iterable, Optional, Tuple, Union, Any

from tqdm import tqdm
from PIL import Image
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

inception_checkpoint = "../../data/models/inception"

class InceptionV3(nn.Module):
    def setup(self):
        self.params_dict = pickle.load(open(inception_checkpoint, "rb"))

    @nn.compact
    def __call__(self, x, train=True):
        """
        Args:
            x (tensor): Input image, shape [B, H, W, C].
            train (bool): If True, training mode.
            rng (jax.random.PRNGKey): Random seed.
        """
        x = self._transform_input(x)
        x = BasicConv2d(
            out_channels=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get_from_dict(self.params_dict, 'Conv2d_1a_3x3'),
            dtype=self.dtype
        )(x, train)

        x = BasicConv2d(out_channels=32,
                        kernel_size=(3, 3),
                        params_dict=get_from_dict(self.params_dict, 'Conv2d_2a_3x3'),
                        dtype=self.dtype)(x, train)
        
        x = BasicConv2d(out_channels=64,
                        kernel_size=(3, 3),
                        padding=((1, 1), (1, 1)),
                        params_dict=get_from_dict(self.params_dict, 'Conv2d_2b_3x3'),
                        dtype=self.dtype)(x, train)
        
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = BasicConv2d(
            out_channels=80,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'Conv2d_3b_1x1'),
            dtype=self.dtype
        )(x, train)
        
        x = BasicConv2d(
            out_channels=192,
            kernel_size=(3, 3),
            params_dict=get_from_dict(self.params_dict, 'Conv2d_4a_3x3'),
            dtype=self.dtype
        )(x, train)
        
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = InceptionA(
            pool_features=32,
            params_dict=get_from_dict(self.params_dict, 'Mixed_5b'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionA(
            pool_features=64,
            params_dict=get_from_dict(self.params_dict, 'Mixed_5c'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionA(
            pool_features=64,
            params_dict=get_from_dict(self.params_dict, 'Mixed_5d'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionB(
            params_dict=get_from_dict(self.params_dict, 'Mixed_6a'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionC(
            channels_7x7=128,
            params_dict=get_from_dict(self.params_dict, 'Mixed_6b'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionC(
            channels_7x7=160,
            params_dict=get_from_dict(self.params_dict, 'Mixed_6c'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionC(
            channels_7x7=160,
            params_dict=get_from_dict(self.params_dict, 'Mixed_6d'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionC(
            channels_7x7=192,
            params_dict=get_from_dict(self.params_dict, 'Mixed_6e'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionD(
            params_dict=get_from_dict(self.params_dict, 'Mixed_7a'),
            dtype=self.dtype
        )(x, train)
        
        x = InceptionE(
            nn.avg_pool, params_dict=get_from_dict(self.params_dict, 'Mixed_7b'),
            dtype=self.dtype
        )(x, train)

        # Following the implementation by @mseitzer, we use max pooling instead
        # of average pooling here.
        # See: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py#L320
        x = InceptionE(
            nn.max_pool, params_dict=get_from_dict(self.params_dict, 'Mixed_7c'),
            dtype=self.dtype
        )(x, train)
        x = jnp.mean(x, axis=(1, 2), keepdims=True)
        return x

class Dense(nn.Module):
    features: int
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=self.features,
            kernel_init= jnp.array(self.params_dict['kernel']),
            bias_init= jnp.array(self.params_dict['bias'])
        )(x)

        return x

class BasicConv2d(nn.Module):
    out_channels: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    strides: Optional[Iterable[int]]=(1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]]='valid'
    use_bias: bool=False
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(
            features=self.out_channels,        
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init= jnp.array(self.params_dict['conv']['kernel']),
            bias_init= jnp.array(self.params_dict['conv']['bias']),
            dtype=self.dtype)(x)
        
        x = nn.BatchNorm(
            epsilon=0.001,
            momentum=0.1,
            bias_init = jnp.array(self.params_dict['bn']['bias']),
            scale_init= jnp.array(self.params_dict['bn']['scale']),
            mean_init= jnp.array(self.params_dict['bn']['mean']),
            var_init= jnp.array(self.params_dict['bn']['var']),
            use_running_average=not train,
            dtype=self.dtype
        )(x)

        x = nn.relu(x)
        return x
    
class InceptionA(nn.Module):
    pool_features: int
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch1x1 = BasicConv2d(
            out_channels=64,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch1x1'),
            dtype=self.dtype
        )(x, train)

        branch5x5 = BasicConv2d(
            out_channels=48,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch5x5_1'),
            dtype=self.dtype
        )(x, train)

        branch5x5 = BasicConv2d(
            out_channels=64,
            kernel_size=(5, 5),
            padding=((2, 2), (2, 2)),
            params_dict=get_from_dict(self.params_dict, 'branch5x5_2'),
            dtype=self.dtype
        )(branch5x5, train)

        branch3x3dbl = BasicConv2d(
            out_channels=64,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_1'),
            dtype=self.dtype
        )(x, train)

        branch3x3dbl = BasicConv2d(
            out_channels=96,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_2'),
            dtype=self.dtype
        )(branch3x3dbl, train)

        branch3x3dbl = BasicConv2d(
            out_channels=96,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_3'),
            dtype=self.dtype
        )(branch3x3dbl, train)

        branch_pool = nn.avg_pool(
            x, 
            window_shape=(3, 3), 
            strides=(1, 1), 
            padding=((1, 1), (1, 1))
        )

        branch_pool = BasicConv2d(
            out_channels=self.pool_features,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch_pool'),
            dtype=self.dtype
        )(branch_pool, train)
        
        output = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=-1)
        return output
    
class InceptionB(nn.Module):
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch3x3 = BasicConv2d(
            out_channels=384,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get_from_dict(self.params_dict, 'branch3x3'),
            dtype=self.dtype
        )(x, train)

        branch3x3dbl = BasicConv2d(
            out_channels=64,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_1'),
            dtype=self.dtype
        )(x, train)

        branch3x3dbl = BasicConv2d(
            out_channels=96,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_2'),
            dtype=self.dtype
        )(branch3x3dbl, train)

        branch3x3dbl = BasicConv2d(
            out_channels=96,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_3'),
            dtype=self.dtype
        )(branch3x3dbl, train)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        output = jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=-1)
        return output
    
class InceptionC(nn.Module):
    channels_7x7: int
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch1x1 = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch1x1'),
            dtype=self.dtype
        )(x, train)
            
        branch7x7 = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch7x7_1'),
            dtype=self.dtype
        )(x, train)

        branch7x7 = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 7),
            padding=((0, 0), (3, 3)),
            params_dict=get_from_dict(self.params_dict, 'branch7x7_2'),
            dtype=self.dtype
        )(branch7x7, train)

        branch7x7 = BasicConv2d(
            out_channels=192,
            kernel_size=(7, 1),
            padding=((3, 3), (0, 0)),
            params_dict=get_from_dict(self.params_dict, 'branch7x7_3'),
            dtype=self.dtype
        )(branch7x7, train)

        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch7x7dbl_1'),
            dtype=self.dtype
        )(x, train)
        
        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(7, 1),
            padding=((3, 3), (0, 0)),
            params_dict=get_from_dict(self.params_dict, 'branch7x7dbl_2'),
            dtype=self.dtype
        )(branch7x7dbl, train)

        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 7),
            padding=((0, 0), (3, 3)),
            params_dict=get_from_dict(self.params_dict, 'branch7x7dbl_3'),
            dtype=self.dtype
        )(branch7x7dbl, train)
        
        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(7, 1),
            padding=((3, 3), (0, 0)),
            params_dict=get_from_dict(self.params_dict, 'branch7x7dbl_4'),
            dtype=self.dtype
        )(branch7x7dbl, train)
        
        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 7),
            padding=((0, 0), (3, 3)),
            params_dict=get_from_dict(self.params_dict, 'branch7x7dbl_5'),
            dtype=self.dtype
        )(branch7x7dbl, train)

        branch_pool = nn.avg_pool(
            x, 
            window_shape=(3, 3), 
            strides=(1, 1), 
            padding=((1, 1), (1, 1))
        )

        branch_pool = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch_pool'),
            dtype=self.dtype
        )(branch_pool, train)
        
        output = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=-1)
        return output
    
class InceptionD(nn.Module):
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch3x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch3x3_1'),
            dtype=self.dtype
        )(x, train)

        branch3x3 = BasicConv2d(
            out_channels=320,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get_from_dict(self.params_dict, 'branch3x3_2'),
            dtype=self.dtype
        )(branch3x3, train)
            
        branch7x7x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch7x7x3_1'),
            dtype=self.dtype
        )(x, train)
        
        branch7x7x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 7),
            padding=((0, 0), (3, 3)),
            params_dict=get_from_dict(self.params_dict, 'branch7x7x3_2'),
            dtype=self.dtype
        )(branch7x7x3, train)
        
        branch7x7x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(7, 1),
            padding=((3, 3), (0, 0)),
            params_dict=get_from_dict(self.params_dict, 'branch7x7x3_3'),
            dtype=self.dtype
        )(branch7x7x3, train)
        
        branch7x7x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get_from_dict(self.params_dict, 'branch7x7x3_4'),
            dtype=self.dtype
        )(branch7x7x3, train)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        
        output = jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=-1)
        return output
    
class InceptionE(nn.Module):
    pooling: Callable
    params_dict: dict=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch1x1 = BasicConv2d(
            out_channels=320,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch1x1'),
            dtype=self.dtype
        )(x, train)
          
        branch3x3 = BasicConv2d(out_channels=384,
                                kernel_size=(1, 1),
                                params_dict=get_from_dict(self.params_dict, 'branch3x3_1'),
                                dtype=self.dtype)(x, train)
        
        branch3x3_a = BasicConv2d(
            out_channels=384,
            kernel_size=(1, 3),
            padding=((0, 0), (1, 1)),
            params_dict=get_from_dict(self.params_dict, 'branch3x3_2a'),
            dtype=self.dtype
        )(branch3x3, train)
        
        branch3x3_b = BasicConv2d(
            out_channels=384,
            kernel_size=(3, 1),
            padding=((1, 1), (0, 0)),
            params_dict=get_from_dict(self.params_dict, 'branch3x3_2b'),
            dtype=self.dtype
        )(branch3x3, train)
        
        branch3x3 = jnp.concatenate((branch3x3_a, branch3x3_b), axis=-1)

        branch3x3dbl = BasicConv2d(
            out_channels=448,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_1'),
            dtype=self.dtype
        )(x, train)
        
        branch3x3dbl = BasicConv2d(
            out_channels=384,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_2'),
            dtype=self.dtype
        )(branch3x3dbl, train)
        
        branch3x3dbl_a = BasicConv2d(
            out_channels=384,
            kernel_size=(1, 3),
            padding=((0, 0), (1, 1)),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_3a'),
            dtype=self.dtype
        )(branch3x3dbl, train)
        
        branch3x3dbl_b = BasicConv2d(
            out_channels=384,
            kernel_size=(3, 1),
            padding=((1, 1), (0, 0)),
            params_dict=get_from_dict(self.params_dict, 'branch3x3dbl_3b'),
            dtype=self.dtype
        )(branch3x3dbl, train)
        
        branch3x3dbl = jnp.concatenate((branch3x3dbl_a, branch3x3dbl_b), axis=-1)

        branch_pool = self.pooling(
            x, 
            window_shape=(3, 3), 
            strides=(1, 1), 
            padding=((1, 1), (1, 1))
        )

        branch_pool = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get_from_dict(self.params_dict, 'branch_pool'),
            dtype=self.dtype
        )(branch_pool, train)
        
        output = jnp.concatenate((branch1x1, branch3x3, branch3x3dbl, branch_pool), axis=-1)
        return output
    
def get_from_dict(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]

def compute_statistics(
    path, 
    params, 
    apply_fn, 
    preprocessing_fn, 
    batch_size=1, 
    image_size=None
):

    idg = ImageDataGenerator(preprocessing_function = preprocessing_fn)
    image_iterator = idg.flow_from_directory(
        path, 
        target_size = image_size, 
        batch_size = batch_size,
        color_mode = 'rgb',
        classes = ['']
    )

    # What is act?
    activations = []
    for _ in tqdm(range(len(image_iterator))):
        x = image_iterator.next()
        x = jnp.asarray(x)
        x = 2 * x - 1
        predictions = apply_fn(params, jax.lax.stop_gradient(x))
        activations.append(predictions.squeeze(axis=1).squeeze(axis=1))
    activations = jnp.concatenate(activations, axis=0)

    mu = jnp.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    pass

