import argparse
import shutil
import os
from tqdm import tqdm

dataset_path = '../../heightmaps/uncorrupted_split_heightmaps_second_pass'
coordinate_list = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'Path to dataset.'
    parser.add_argument('--dataset_path', default=dataset_path, type=str, help=help_text)

    assert os.path.exists(parser.parse_args().dataset_path), 'Dataset path does not exist'

    dataset_list = os.listdir(parser.parse_args().dataset_path)
    for image in tqdm(dataset_list):
        coordinate_string = image[:8]
        destination_path = os.path.join(dataset_path, coordinate_string)
        if coordinate_string not in coordinate_list:
            coordinate_list.append(coordinate_string)
            os.makedirs(os.path.join(dataset_path, coordinate_string))
        shutil.move(os.path.join(dataset_path, image), destination_path)
        