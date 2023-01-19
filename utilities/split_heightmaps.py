from split_image import split_image
import os

input_directory = 'D:\heightmap_pngs/'
output_directory = 'D:\split_heightmaps/'

file_list = os.listdir(input_directory)

for i in range(len(file_list)):
    split_image(input_directory + file_list[i], 10, 10, should_square=True, should_cleanup=False, output_dir=output_directory)
