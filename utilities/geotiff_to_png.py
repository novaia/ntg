import rasterio
import matplotlib.pyplot as plt
import os

input_directory = '/Users/Hayden/Desktop/geotiffs/'
output_directory = 'D:\heightmap_pngs/'

file_list = os.listdir(input_directory)

for i in range(len(file_list)):
    image = rasterio.open(input_directory + file_list[i])
    plt.imsave(output_directory + file_list[i][0:-4] + '.png', image.read(1), cmap = 'gray')