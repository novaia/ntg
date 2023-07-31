"""
Frechet Inception Distance.
Mostly taken from: https://github.com/matthias-wright/jax-fid/tree/main
License: https://github.com/matthias-wright/jax-fid/blob/main/LICENSE
The code in this file was modified from the original.
"""
import jax
import functools
from tqdm import tqdm
import os
import numpy as np
import jax.numpy as jnp
from fid.inception import InceptionV3
import scipy

def compute_statistics_with_mmap(
    params, apply_fn, num_batches, batch_size, 
    get_batch_fn, filename, dtype, num_activations
):    
    activation_dim = 2048
    mm = np.memmap(filename, dtype=dtype, mode='w+', shape=(num_activations, activation_dim))

    activation_sum = np.zeros((activation_dim))
    for i in tqdm(range(num_batches)):
        x = get_batch_fn()
        x = np.asarray(x)
        x = 2 * x - 1
        activation_batch = apply_fn(params, jax.lax.stop_gradient(x))
        activation_batch = activation_batch.squeeze(axis=1).squeeze(axis=1)

        current_batch_size = activation_batch.shape[0]
        start_index = i * batch_size
        end_index = start_index + current_batch_size
        mm[start_index : end_index] = activation_batch

        activation_sum += activation_batch.sum(axis=0)

    mu = activation_sum / num_activations
    sigma = np.cov(mm, rowvar=False)

    return mu, sigma

def compute_statistics(params, apply_fn, num_batches, get_batch_fn):
    activations = []

    for _ in tqdm(range(num_batches)):
        x = get_batch_fn()
        x = np.asarray(x)
        x = 2 * x - 1
        pred = apply_fn(params, jax.lax.stop_gradient(x))
        activations.append(pred.squeeze(axis=1).squeeze(axis=1))
    activations = jnp.concatenate(activations, axis=0)
    print(f'num activations orig: {activations.shape[0]}')

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def load_statistics(path):
    stats = np.load(path)
    mu, sigma = stats["mu"], stats["sigma"]
    return mu, sigma

def save_statistics(path, mu, sigma):
    np.savez(path, mu=mu, sigma=sigma)

# Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_1d(sigma1)
    sigma2 = np.atleast_1d(sigma2)

    assertion_text = f'mu shapes must be the same but are {mu1.shape} and {mu2.shape}'
    assert mu1.shape == mu2.shape, assertion_text
    assertion_text = f'sigma shapes must be the same but are {sigma1.shape} and {sigma2.shape}'
    assert sigma1.shape == sigma2.shape, assertion_text

    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_inception_model():
    rng = jax.random.PRNGKey(0)
    model = InceptionV3()
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))
    apply_fn = jax.jit(functools.partial(model.apply, train=False))
    return params, apply_fn

def preprocessing_function(image):
    image = image.astype(float) / 255
    return image