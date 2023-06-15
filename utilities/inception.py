"""
Mostly taken from: https://github.com/matthias-wright/jax-fid/tree/main
License: https://github.com/matthias-wright/jax-fid/blob/main/LICENSE
The code in this file was modified from the original.
"""


import flax.linen as nn
import jax.numpy as jnp
import jax
from jax import lax
from jax.nn import initializers
import pickle
import functools
from typing import Callable, Iterable, Optional, Tuple, Union, Any
import os

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any

class InceptionV3(nn.Module):
    checkpoint_path: str='../data/inception_v3_weights_fid_old.pickle'
    dtype: str='float32'

    def setup(self):
        assert os.path.isfile(self.checkpoint_path), "Inception checkpoint not found"
        self.params_dict = pickle.load(open(self.checkpoint_path, "rb"))

    @nn.compact
    def __call__(self, x, train=True):
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
            kernel_init = lambda *_ : jnp.array(get_from_dict(self.params_dict, 'kernel')),
            bias_init = lambda *_ :  jnp.array(get_from_dict(self.params_dict, 'bias'))
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
            features = self.out_channels,        
            kernel_size = self.kernel_size,
            strides = self.strides,
            padding = self.padding,
            use_bias = self.use_bias,
            kernel_init = lambda *_ : jnp.array(get_from_dict(self.params_dict['conv'], 'kernel')),
            bias_init = lambda *_ : jnp.array(get_from_dict(self.params_dict['conv'], 'bias')),
            dtype=self.dtype
        )(x)
        
        x = BatchNorm(
            epsilon=0.001,
            momentum=0.1,
            bias_init = lambda *_ :  jnp.array(self.params_dict['bn']['bias']),
            scale_init = lambda *_ :  jnp.array(self.params_dict['bn']['scale']),
            mean_init = lambda *_ :  jnp.array(self.params_dict['bn']['mean']),
            var_init = lambda *_ :  jnp.array(self.params_dict['bn']['var']),
            use_running_average=not train,
            dtype=self.dtype
        )(x)

        x = nn.relu(x)
        return x
    
# Taken from: https://github.com/google/flax/blob/master/flax/linen/normalization.py
class BatchNorm(nn.Module):
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    mean_init: Callable[[Shape], Array] = lambda s: jnp.zeros(s, jnp.float32)
    var_init: Callable[[Shape], Array] = lambda s: jnp.ones(s, jnp.float32)
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        use_running_average = nn.module.merge_param(
            'use_running_average', 
            self.use_running_average, 
            use_running_average
        )
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # see NOTE above on initialization behavior
        initializing = self.is_mutable_collection('params')

        ra_mean = self.variable(
            'batch_stats', 
            'mean',
            self.mean_init,
            reduced_feature_shape
        )
        ra_var = self.variable(
            'batch_stats', 
            'var',
            self.var_init,
            reduced_feature_shape
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(
                        concatenated_mean,
                        axis_name=self.axis_name,
                        axis_index_groups=self.axis_index_groups), 2)
            var = mean2 - lax.square(mean)

            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param(
                'scale',
                self.scale_init,
                reduced_feature_shape
            ).reshape(feature_shape)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param(
                'bias',
                self.bias_init,
                reduced_feature_shape
            ).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)
    
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
    
def absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])

def get_from_dict(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]