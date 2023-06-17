"""
Frechet Inception Distance.
Mostly taken from: https://github.com/matthias-wright/jax-fid/tree/main
License: https://github.com/matthias-wright/jax-fid/blob/main/LICENSE
The code in this file was modified from the original.
"""
import sys
sys.path.append('../')

import jax.numpy as jnp
import jax
import functools
from tqdm import tqdm
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from fid import fid_inception
import scipy
import argparse

# temp
from PIL import Image

def compute_statistics(path, params, apply_fn, batch_size=1, img_size=None):
    images = []
    for f in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, f))
        # Convert if only a single channel.
        if img.mode == "L":
            img = img.convert("RGB")
        # Resize if not the right size.
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

def save_statistics(path, mu, sigma):
    np.savez(path, mu=mu, sigma=sigma)

# Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
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
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def _precompute_and_save_statistics(args, params, apply_fn):
    error_text = 'img_dir must be specified if precompute_stats is True'
    assert args.img_dir is not None, error_text
    error_text = 'out_dir must be specified if precompute_stats is True'
    assert args.out_dir is not None, error_text 
    
    mu, sigma = compute_statistics(
        args.img_dir, 
        params, 
        apply_fn, 
        args.batch_size, 
        args.img_size
    )

    os.makedirs(args.out_dir, exist_ok=True)
    save_statistics(os.path.join(args.out_dir, args.out_name), mu=mu, sigma=sigma)
    print(
        'Saved pre-computed statistics at:', 
        os.path.join(args.out_dir, args.out_name + '.npz')
    )

def _get_statistics_and_compute_fid(args, params, apply_fn):
    error_text = 'path1 must be specified if precompute_stats is False'
    assert args.path1 is not None, error_text
    error_text = 'path2 must be specified if precompute_stats is False'
    assert args.path2 is not None, error_text

    if args.path1.endswith('.npz'):
        mu1, sigma1 = load_statistics(args.path1)
    else:
        mu1, sigma1 = compute_statistics(
            args.path1, 
            params, 
            apply_fn, 
            args.batch_size, 
            args.img_size
        )

    if args.path2.endswith('.npz'):
        mu2, sigma2 = load_statistics(args.path2)
    else:
        mu2, sigma2 = compute_statistics(
            args.path2, 
            params, 
            apply_fn, 
            args.batch_size, 
            args.img_size
        )

    fid = compute_frechet_distance(mu1, mu2, sigma1, sigma2)
    print('FID:', fid)

def preprocessing_function(image):
    image = image.astype(float) / 255
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'Path to image directory or .npz file containing pre-computed statistics.'
    parser.add_argument('--path1', type=str, help=help_text)
    help_text = 'Path to image directory or .npz file containing pre-computed statistics.'
    parser.add_argument('--path2', type=str, help=help_text)
    help_text = 'Batch size per device for computing the Inception activations.'
    parser.add_argument('--batch_size', type=int, default=50, help=help_text)
    help_text = 'Resize images to this size. The format is (height, width).'
    parser.add_argument('--img_size', type=int, nargs=2, help=help_text)
    help_text = 'If True, pre-compute statistics for given image directory.'
    parser.add_argument('--precompute', action='store_true', help=help_text)
    help_text = 'Path to image directory for pre-computing statistics.'
    parser.add_argument('--img_dir', type=str, help=help_text)
    help_text = 'Path where pre-computed statistics are stored.'
    parser.add_argument('--out_dir', type=str, help=help_text)
    help_text = 'Name of outputted statistics file.'
    parser.add_argument('--out_name', type=str, default='stats', help='Name of dataset')
    args = parser.parse_args()

    rng = jax.random.PRNGKey(0)
    model = fid_inception.InceptionV3()
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))
    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    if args.precompute:
        _precompute_and_save_statistics(args, params, apply_fn)
    else:
        _get_statistics_and_compute_fid(args, params, apply_fn)