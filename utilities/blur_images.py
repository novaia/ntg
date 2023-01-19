import os
import cv2

input_directory = '/Users/Hayden/Desktop/ml/split_heightmaps/'
output_directory = '../../Desktop/ml/blurred_heightmaps/'

file_list = os.listdir(input_directory)

for i in range(len(file_list)):
    image = cv2.imread('../../Desktop/ml/split_heightmaps/' + file_list[i])
    resized_image = cv2.resize(image, (256, 256))
    blurred_image = cv2.blur(resized_image, (40, 40))
    cv2.imwrite(output_directory + 'blurred-' + file_list[i], blurred_image)