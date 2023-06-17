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

# temp
from PIL import Image
#import intest

def test_compute_statistics(path, params, apply_fn, batch_size=1, img_size=None):
    if path.endswith(".npz"):
        stats = np.load(path)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma

    images = []
    for f in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, f))
        # convert if only a single channel
        if img.mode == "L":
            img = img.convert("RGB")
        # resize if not the right size
        if img_size is not None and img.size[:2] != img_size:
            img = img.resize(
                size=(img_size[0], img_size[1]),
                resample=Image.BILINEAR,
            )
        img = np.array(img) / 255.0
        images.append(img)

    num_batches = int(len(images) // batch_size)
    act = []
    for i in tqdm(range(num_batches)):
        x = images[i * batch_size : i * batch_size + batch_size]
        x = np.asarray(x)
        x = 2 * x - 1
        pred = apply_fn(params, jax.lax.stop_gradient(x))
        act.append(pred.squeeze(axis=1).squeeze(axis=1))
    act = jnp.concatenate(act, axis=0)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

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
    rng = jax.random.PRNGKey(0)
    #model = intest.InceptionV3(pretrained=True)
    model = fid_inception.InceptionV3()
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))
    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    mu1, sigma1 = test_compute_statistics(
        #'../data/dataset_info/fid_test1_stats.npz',
        '../../fid_test1',
        params,
        apply_fn,
        batch_size=50
    )
    mu2, sigma2 = test_compute_statistics(
        #'../data/dataset_info/fid_test2_stats.npz',
        '../../fid_test2',
        params,
        apply_fn,
        batch_size=50
    )

    fid = test_function(mu1, mu2, sigma1, sigma2)
    print('FID:', fid)