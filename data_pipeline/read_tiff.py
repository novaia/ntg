import os, argparse
from tifffile import imread, TiffFile
from PIL import Image
import zarr
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    input_paths = os.listdir(args.input_path)
    for path in input_paths:
        tif = TiffFile(os.path.join(args.input_path, path))
        store = tif.aszarr(key=0)
        z = zarr.open(store, mode='r')
        heightmap = np.array(z, np.float32)
        heightmap = np.clip(heightmap, 0.0, heightmap.max())
        heightmap = heightmap / heightmap.max()
        heightmap = np.array(heightmap * 255, np.uint8)
        heightmap_image = Image.fromarray(heightmap)
        heightmap_image.save(os.path.join(args.output_path, f'{path[:-3]}png'))
        store.close()
        heightmap_image.close()

if __name__ == '__main__':
    main()
