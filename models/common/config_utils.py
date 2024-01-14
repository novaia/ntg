from jax import numpy as jnp
from flax import linen as nn
import optax
import argparse

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

def load_optimizer(config:dict, learning_rate):
    optimizer_name = config['optimizer']
    if optimizer_name == 'sgd':
        return optax.sgd(
            learning_rate=learning_rate, momentum=config['sgd_momentum'], nesterov=config['sgd_nesterov']
        )
    elif optimizer_name == 'adam':
        return optax.adam(
            learning_rate=learning_rate, b1=config['adam_b1'], b2=config['adam_b2'], eps=config['adam_eps'], 
            eps_root=config['adam_eps_root']
        )
    elif optimizer_name == 'adamw':
         return optax.adamw(
            learning_rate=learning_rate, b1=config['adam_b1'], b2=config['adam_b2'], eps=config['adam_eps'], 
            eps_root=config['adam_eps_root'], weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(
            f'Invalid optimizer: {optimizer_name}.',
            'Must be one of the following: sgd, adam, adamw.'
        )

def parse_args(default_run_dir:str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--wandb', type=int, choices=[0, 1], default=1)
    parser.add_argument('--epochs_between_previews', type=int, default=1)
    parser.add_argument('--steps_between_wandb_logs', type=int, default=200)
    parser.add_argument('--save_checkpoints', type=int, choices=[0, 1], default=1)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=default_run_dir)
    parser.add_argument('--tabulate', type=int, choices=[0, 1], default=0)
    return parser.parse_args()
