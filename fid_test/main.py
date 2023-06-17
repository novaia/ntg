import jax
import jax.numpy as jnp
import flax
import numpy as np
import argparse
import functools
import os

import inception
import fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, help='Path to image directory or .npz file containing pre-computed statistics.')
    parser.add_argument('--path2', type=str, help='Path to image directory or .npz file containing pre-computed statistics.')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size per device for computing the Inception activations.')
    parser.add_argument('--img_size', type=int, nargs=2, help='Resize images to this size. The format is (height, width).')
    parser.add_argument('--precompute', action='store_true', help='If True, pre-compute statistics for given image directory.')
    parser.add_argument('--img_dir', type=str, help='Path to image directory for pre-computing statistics.')
    parser.add_argument('--out_dir', type=str, help='Path where pre-computed statistics are stored.')
    args = parser.parse_args()

    rng = jax.random.PRNGKey(0)

    model = inception.InceptionV3(pretrained=True)
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))

    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    if args.precompute:
        assert args.img_dir is not None, 'img_dir must be specified if precompute_stats is True'
        assert args.out_dir is not None, 'out_dir must be specified if precompute_stats is True'
        mu, sigma = fid.compute_statistics(args.img_dir, params, apply_fn, args.batch_size, args.img_size)
        os.makedirs(args.out_dir, exist_ok=True)
        np.savez(os.path.join(args.out_dir, 'stats'), mu=mu, sigma=sigma)
        print('Saved pre-computed statistics at:', os.path.join(args.out_dir, 'stats.npz'))
    else:
        assert args.path1 is not None, 'path1 must be specified'
        assert args.path2 is not None, 'path2 must be specified'
        assert os.path.isdir(args.path1) or args.path1.endswith('.npz'), 'path1 must be a directory or an .npz file'
        assert os.path.isdir(args.path2) or args.path2.endswith('.npz'), 'path2 must be a directory or an .npz file'
        mu1, sigma1 = fid.compute_statistics(args.path1, params, apply_fn, args.batch_size, args.img_size)
        mu2, sigma2 = fid.compute_statistics(args.path2, params, apply_fn, args.batch_size, args.img_size)
        

        fid_score = fid.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
        print('Fid:', fid_score)


if __name__ == '__main__':
    main()