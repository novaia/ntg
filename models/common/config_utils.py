from jax import numpy as jnp
from flax import linen as nn

ACTIVATION_FN_MAP = {'gelu': nn.gelu, 'silu': nn.silu}
DTYPE_MAP = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16}

def load_activation_fn(activation_fn_name:str):
    assert activation_fn_name in ACTIVATION_FN_MAP.keys(), (
        f'Invalid activation function: {ACTIVATION_FN_MAP}. ',
        f'Must be one of the following: {list(ACTIVATION_FN_MAP.keys())}.'
    )
    return ACTIVATION_FN_MAP[activation_fn_name]

def load_dtype(dtype_name:str):
    assert dtype_name in DTYPE_MAP.keys(), (
        f'Invalid dtype: {dtype_name}. Must be one of the following: {list(DTYPE_MAP.keys())}.'
    )
    return DTYPE_MAP[dtype_name]