# This script is meant to be used for heightmaps that have already been
# organized into longitude and latitude folders by 
# organize_heightmaps_into_folders.py. This script will remove redundant
# .png postfixes from the heightmap images and rename them to their slice ID.
# The slice ID is the 1-dimensional coordinate of the heightmap with respect to
# its latitude and longitude. There were a maximum of 100 slices per
# latitude/longitude, so the slice ID is a number between 0 and 99.

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_text = 'Path to directory containing heightmaps.'
    parser.add_argument('--directory', type=str, help=help_text)
    help_text = 'Postfix to remove from heightmap filenames.'
    parser.add_argument('--postfix_to_remove', type=str, default='.png.png.png', help=help_text)
    args = parser.parse_args()

    assert os.path.exists(args.directory), 'Directory does not exist'

    for folder in os.listdir(args.directory):
        folder_path = os.path.join(args.directory, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(args.postfix_to_remove):
                new_filename = filename[:-len(args.postfix_to_remove)]
                slice_id = new_filename.split('_')[-1]
                if slice_id == 'squared':
                    new_filename = 'whole'
                else:
                    new_filename = str(slice_id)
                new_filename += '.png'
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)