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
#from fid import fid_inception
import fid_inception
import scipy
import argparse
import mmap

# temp
from PIL import Image

def compute_statistics_mmapped(
    params, apply_fn, num_batches, batch_size, get_batch_fn, filename, dtype
):
    assert num_batches > 0, "num_batches must be greater than 0"
    assert batch_size > 0, "batch_size must be greater than 0"
    
    activation_dim = 2048
    #num_activations = num_batches * batch_size
    num_activations = 0
    # _b suffix means the size is specifically the size of something in number of bytes,
    # as opposed to the size of something in number of elements.
    dtype_size_b = np.dtype(dtype).itemsize
    activation_size_b = dtype_size_b * activation_dim
    file_size_b = activation_size_b * num_batches * batch_size

    with open(filename, "w+b") as f:
        f.write(b"\0" * file_size_b)
        f.flush()
        mm = mmap.mmap(f.fileno(), file_size_b)

        #activation_sum = np.zeros((activation_dim), dtype=dtype)
        activation_sum = np.zeros((activation_dim))
        for i in tqdm(range(num_batches)):
            x = get_batch_fn()
            #x = np.asarray(x, dtype=dtype)
            #x = np.asarray(x)
            x = 2 * x - 1
            activation_batch = apply_fn(params, jax.lax.stop_gradient(x))
            activation_batch = activation_batch.squeeze(axis=1).squeeze(axis=1)

            for k, activation in enumerate(activation_batch):
                activation_sum += activation
                num_activations += 1
                start_index = (i + k) * activation_size_b
                end_index = (i + k + 1) * activation_size_b
                mm[start_index : end_index] = activation.tobytes()

        print(f'activation_sum dtype: {activation_sum.dtype}')
        mu = activation_sum / num_activations
        print(f'mu dtype: {mu.dtype}')
        print(f'num activations mmap: {num_activations}')

        '''
        sigma = np.zeros((activation_dim, activation_dim), dtype=dtype)
        observations = np.zeros((2, activation_dim), dtype=dtype)
        for a_id in tqdm(range(activation_dim)):
            for b_id in range(activation_dim):

                # Load the observations for variables a and b.
                for i in range(num_activations):
                    a_offset = i * activation_size_b + a_id * dtype_size_b
                    observations[0][i] = np.frombuffer(
                        mm[a_offset : a_offset + dtype_size_b], dtype=dtype
                    )
                    b_offset = i * activation_size_b + b_id * dtype_size_b
                    observations[1][i] = np.frombuffer(
                        mm[b_offset : b_offset + dtype_size_b], dtype=dtype
                    )
                # Only want cov(a, b), not the whole matrix.
                sigma[a_id][b_id] = np.cov(observations, bias=True)[0][1]
        '''

    return mu, np.zeros((1))

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

def _get_directory_iterator(path, args, data_generator):
    directory_iterator = data_generator.flow_from_directory(
        path,
        target_size = args.img_size,
        batch_size = args.batch_size,
        color_mode = 'rgb',
        classes = ['']
    )
    return directory_iterator

def _precompute_and_save_statistics(args, params, apply_fn, data_generator):
    error_text = 'img_dir must be specified if precompute_stats is True'
    assert args.img_dir is not None, error_text
    error_text = 'out_dir must be specified if precompute_stats is True'
    assert args.out_dir is not None, error_text 

    directory_iterator = _get_directory_iterator(args.img_dir, args, data_generator)

    mu, sigma = compute_statistics(
        params, 
        apply_fn, 
        num_batches = len(directory_iterator),
        get_batch_fn = lambda: directory_iterator.next()[0]
    )

    os.makedirs(args.out_dir, exist_ok=True)
    save_statistics(os.path.join(args.out_dir, args.out_name), mu=mu, sigma=sigma)
    print(
        'Saved pre-computed statistics at:', 
        os.path.join(args.out_dir, args.out_name + '.npz')
    )

def _get_statistics_and_compute_fid(args, params, apply_fn, data_generator):
    error_text = 'path1 must be specified if precompute_stats is False'
    assert args.path1 is not None, error_text
    error_text = 'path2 must be specified if precompute_stats is False'
    assert args.path2 is not None, error_text


    if args.path1.endswith('.npz'):
        mu1, sigma1 = load_statistics(args.path1)
    else:
        directory_iterator1 = _get_directory_iterator(args.path1, args, data_generator)
        mu1, sigma1 = compute_statistics(
            params, 
            apply_fn, 
            num_batches = len(directory_iterator1),
            get_batch_fn = lambda: directory_iterator1.next()[0]
        )

    if args.path2.endswith('.npz'):
        mu2, sigma2 = load_statistics(args.path2)
    else:
        directory_iterator2 = _get_directory_iterator(args.path2, args, data_generator)
        mu2, sigma2 = compute_statistics(
            params, 
            apply_fn, 
            num_batches = len(directory_iterator2),
            get_batch_fn = lambda: directory_iterator2.next()[0]
        )

    fid = compute_frechet_distance(mu1, mu2, sigma1, sigma2)
    print('FID:', fid)

def get_inception_model():
    rng = jax.random.PRNGKey(0)
    model = fid_inception.InceptionV3()
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))
    apply_fn = jax.jit(functools.partial(model.apply, train=False))
    return params, apply_fn

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

    params, apply_fn = get_inception_model()
    idg = ImageDataGenerator(preprocessing_function = preprocessing_function)

    if args.precompute:
        _precompute_and_save_statistics(args, params, apply_fn, idg)
    else:
        _get_statistics_and_compute_fid(args, params, apply_fn, idg)