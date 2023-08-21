import os
import random
import shutil
import argparse
from tqdm import tqdm

dataset_path = '../../heightmaps/world-heightmaps-01/train'
output_path = '../../heightmaps/world-heightmaps-01/test'
test_split = 0.05
file_extension = '.png'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'Path to dataset directory.'
    parser.add_argument('--dataset_path', type=str, default=dataset_path)
    help_text = 'Path to output directory for test set.'
    parser.add_argument('--output_path', type=str, default=output_path)
    help_text = 'Percentage of dataset to be reserved for test set.'
    parser.add_argument('--test_split', type=float, default=test_split)
    help_text = 'File extension of dataset files.'
    parser.add_argument('--file_extension', type=str, default=file_extension)
    args = parser.parse_args()

    file_list = []

    for root, dirs, files in os.walk(args.dataset_path):
        for file in files:
            if file.endswith(".png"):
                file_list.append(os.path.join(root, file))
    print(file_list[0])
    print(f'Found {len(file_list)} files.')
    test_set_size = int(len(file_list) * args.test_split)
    print(f'Copying {test_set_size} files to {args.output_path}.')

    selected_files = random.sample(file_list, test_set_size)
    for i in tqdm(range(len(selected_files))):
        file = selected_files[i]
        split_file_path = file.split('/')
        class_folder = split_file_path[-2]
        file_name = split_file_path[-1]

        folder_output_path = os.path.join(args.output_path, class_folder)
        if not os.path.exists(folder_output_path):
            os.makedirs(folder_output_path)

        file_output_path = os.path.join(folder_output_path, file_name)
        shutil.move(file, file_output_path)
