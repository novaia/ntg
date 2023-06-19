import argparse
import shutil
import os
import random
from tqdm import tqdm

dataset_path = '../../heightmaps/uncorrupted_split_heightmaps_second_pass'
out_path = '../../heightmaps/uncorrupted_split_heightmaps_second_pass_eval'
num_samples = 5000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'Path to original dataset.'
    parser.add_argument('--dataset_path', type=str, default=dataset_path, help=help_text)
    help_text = 'Path to output directory.'
    parser.add_argument('--out_path', type=str, default=out_path, help=help_text)
    help_text = 'Number of random samples to take from original dataset.'
    parser.add_argument('--num_samples', type=int, default=num_samples, help=help_text)
    help_text = 'Force script to run even if output directory is not empty.'
    parser.add_argument('--force_non_empty', type=bool, default=False, help=help_text)
    args = parser.parse_args()

    if os.path.exists(args.out_path) and not args.force_non_empty:
        out_path_is_empty = len(os.listdir(args.out_path)) == 0
        assertion_text = 'Out path is not empty and --force_non_empty is False'
        assert out_path_is_empty, assertion_text
    else:
        os.makedirs(args.out_path)

    assertion_text = 'Dataset path does not exist'
    assert os.path.exists(args.dataset_path), assertion_text
    
    dataset_list = os.listdir(args.dataset_path)
    assertion_text = 'Dataset has fewer files than num_samples'
    assert not len(dataset_list) < args.num_samples, assertion_text

    random_samples = random.sample(dataset_list, args.num_samples)
    print('Randomly sampled', args.num_samples, 'files from:', args.dataset_path)
    print('Copying samples to:', args.out_path)
    for sample in tqdm(random_samples):
        shutil.copy(os.path.join(args.dataset_path, sample), args.out_path)