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
):
    if os.path.isfile(statistics_path):
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
        np.savez(statistics_path, mu=mu, sigma=sigma)
        print('FID statistics saved to:', statistics_path)
        return mu, sigma

def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    pass

def preprocessing_function(image):
    image = image.astype(float) / 255
    return image

if __name__ == '__main__':
    statistics_path = '../data/dataset_info/second_pass_fid_stats.json'
    dataset_path = '../../fid_test1/'

    rng = jax.random.PRNGKey(0)
    model = inception.InceptionV3()
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))
    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    mu, sigma = get_fid_statistics(
        statistics_path,
        dataset_path,
        params,
        apply_fn,
        preprocessing_function,
        image_size=(256, 256)
    )
    print('mu:', mu, 'sigma:', sigma)