"""
Frechet Inception Distance.
Mostly taken from: https://github.com/matthias-wright/jax-fid/tree/main
License: https://github.com/matthias-wright/jax-fid/blob/main/LICENSE
The code in this file was modified from the original.
"""

import jax.numpy as jnp
import jax
import functools
from tqdm import tqdm
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import fid_inception
import json
import scipy

def compute_statistics(
    path, 
    params, 
    apply_fn, 
    preprocessing_fn,
    image_size, 
    batch_size=1 
):
    idg = ImageDataGenerator(preprocessing_function = preprocessing_fn)
    image_iterator = idg.flow_from_directory(
        path, 
        target_size = image_size, 
        batch_size = batch_size,
        color_mode = 'rgb',
        classes = ['']
    )

    activations = []
    for _ in tqdm(range(len(image_iterator))):
        x = jnp.asarray(image_iterator.next()[0])
        x = 2 * x - 1
        predictions = apply_fn(params, jax.lax.stop_gradient(x))
        activations.append(predictions.squeeze(axis=1).squeeze(axis=1))
    activations = jnp.concatenate(activations, axis=0)

    mu = jnp.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    print('mu shape', mu.shape)
    print('sigma shape', sigma.shape)

    return mu, sigma

def load_statistics(path):
    stats = np.load(path)
    mu, sigma = stats["mu"], stats["sigma"]
    return mu, sigma

# Tries to load stastics from a file, if it fails then it computes them.
def get_fid_statistics(
    statistics_path, 
    dataset_path, 
    params, 
    apply_fn, 
    preprocessing_fn,
    image_size,
    save_statistics=False
):
    if statistics_path is not None and os.path.isfile(statistics_path):
        mu, sigma = load_statistics(statistics_path)
        print('FID statistics loaded from:', statistics_path)
        return mu, sigma
    else:
        print('FID statistics not found, computing...')
        mu, sigma = compute_statistics(
            dataset_path, 
            params, 
            apply_fn, 
            preprocessing_fn, 
            image_size
        )
        
        if save_statistics:
            np.savez(statistics_path, mu=mu, sigma=sigma)
            print('FID statistics saved to:', statistics_path)
        
        return mu, sigma

def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_1d(sigma1)
    sigma2 = np.atleast_1d(sigma2)
    
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm(
            (sigma1 + offset).dot(sigma2 + offset), 
            disp=False
        )[0]

    # Numerical error might give slight imaginary component.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def preprocessing_function(image):
    image = image.astype(float) / 255
    return image

if __name__ == '__main__':
    statistics_path = '../data/dataset_info/second_pass_fid_stats.npz'
    dataset_path = '../../fid_test1/'
    second_dataset_path = '../../fid_test2/'

    rng = jax.random.PRNGKey(0)
    model = fid_inception.InceptionV3()
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))
    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    mu1, sigma1 = get_fid_statistics(
        statistics_path,
        dataset_path,
        params,
        apply_fn,
        preprocessing_function,
        image_size=(256, 256),
        save_statistics=True
    )

    mu2, sigma2 = get_fid_statistics(
        'none',
        second_dataset_path,
        params,
        apply_fn,
        preprocessing_function,
        image_size=(256, 256),
        save_statistics=False
    )

    fid = compute_frechet_distance(mu1, mu2, sigma1, sigma2)
    print('FID:', fid)

    