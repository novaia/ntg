import argparse
import os
from tqdm import tqdm

dataset_path = '../../heightmaps/uncorrupted_split_heightmaps_second_pass'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'Path to dataset.'
    parser.add_argument('--dataset_path', default=dataset_path, type=str, help=help_text)

    assert os.path.exists(parser.parse_args().dataset_path), 'Dataset path does not exist'

    dataset_list = os.listdir(parser.parse_args().dataset_path)
    for directory in tqdm(dataset_list):
        directory_path = os.path.join(dataset_path, directory)
        directory_list = os.listdir(directory_path)
        for i, image in enumerate(directory_list):
            old_name = os.path.join(directory_path, image)
            new_name = os.path.join(directory_path, str(i) + '.png')
            os.rename(old_name, new_name)