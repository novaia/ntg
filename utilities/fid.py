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
#import fid_inception
import json
import scipy

# temp
from PIL import Image
import intest

def load_statistics(path):
    stats = np.load(path)
    mu, sigma = stats["mu"], stats["sigma"]
    return mu, sigma

def test_function(mu1, mu2, sigma1, sigma2, eps=1e-6):
    # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
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
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
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
    mu1, sigma1 = load_statistics('../data/dataset_info/fid_test1_stats.npz')
    mu2, sigma2 = load_statistics('../data/dataset_info/fid_test2_stats.npz')

    fid = test_function(mu1, mu2, sigma1, sigma2)
    print('FID:', fid)