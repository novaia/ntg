"""
Frechet Inception Distance.
Mostly taken from: https://github.com/matthias-wright/jax-fid/tree/main
License: https://github.com/matthias-wright/jax-fid/blob/main/LICENSE
The code in this file was modified from the original.
"""
import argparse
import fid
import os
from keras.preprocessing.image import ImageDataGenerator

# Wraps compute statistics functions and decides which one to call based on if mmap is True.
def _compute_statistics_wrapper(
    params, apply_fn, num_batches, get_batch_fn, num_activations, args
):
    if args.mmap:
        mu, sigma = fid.compute_statistics_with_mmap(
            params = params, 
            apply_fn = apply_fn, 
            num_batches = num_batches, 
            batch_size = args.batch_size, 
            get_batch_fn = get_batch_fn,
            filename = args.mmap_filename,
            dtype = 'float32',
            num_activations = num_activations
        )
    else:
        mu, sigma = fid.compute_statistics(params, apply_fn, num_batches, get_batch_fn)
    return mu, sigma

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
    mu, sigma = _compute_statistics_wrapper(
        params = params, 
        apply_fn = apply_fn, 
        num_batches = len(directory_iterator), 
        get_batch_fn = lambda: directory_iterator.next()[0], 
        num_activations = directory_iterator.samples, 
        args = args
    )

    os.makedirs(args.out_dir, exist_ok=True)
    fid.save_statistics(os.path.join(args.out_dir, args.out_name), mu=mu, sigma=sigma)
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
        mu1, sigma1 = fid.load_statistics(args.path1)
    else:
        directory_iterator1 = _get_directory_iterator(args.path1, args, data_generator)
        mu1, sigma1 = _compute_statistics_wrapper(
            params = params, 
            apply_fn = apply_fn, 
            num_batches = len(directory_iterator1), 
            get_batch_fn = lambda: directory_iterator1.next()[0], 
            num_activations = directory_iterator1.samples, 
            args = args
        )

    if args.path2.endswith('.npz'):
        mu2, sigma2 = fid.load_statistics(args.path2)
    else:
        directory_iterator2 = _get_directory_iterator(args.path2, args, data_generator)
        mu2, sigma2 = _compute_statistics_wrapper(
            params = params, 
            apply_fn = apply_fn, 
            num_batches = len(directory_iterator2), 
            get_batch_fn = lambda: directory_iterator2.next()[0], 
            num_activations = directory_iterator2.samples, 
            args = args
        )

    frechet_distance = fid.compute_frechet_distance(mu1, mu2, sigma1, sigma2)
    print('FID:', frechet_distance)

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
    parser.add_argument('--out_name', type=str, default='stats', help=help_text)
    help_text = 'If True, use mmap to compute statistics. Helpful for large datasets.'
    parser.add_argument('--mmap', type=bool, default=True, help=help_text)
    mmap_filename = 'data/temp/mmap_file'
    help_text = 'Name for mmap file. Only used if mmap is True.'
    parser.add_argument('--mmap_filename', type=str, default=mmap_filename, help=help_text)
    args = parser.parse_args()

    params, apply_fn = fid.get_inception_model()
    idg = ImageDataGenerator(preprocessing_function = fid.preprocessing_function)

    if args.precompute:
        _precompute_and_save_statistics(args, params, apply_fn, idg)
    else:
        _get_statistics_and_compute_fid(args, params, apply_fn, idg)