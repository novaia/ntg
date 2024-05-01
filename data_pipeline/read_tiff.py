from tifffile import imread, TiffFile
from PIL import Image
import zarr
import numpy as np

tif = TiffFile('data/example3.tif')
store = tif.aszarr(key=0)
print(store)
z = zarr.open(store, mode="r")
print(z)
heightmap = np.array(z, np.float32)
heightmap = np.clip(heightmap, 0.0, heightmap.max())
print(heightmap.max())
print(heightmap.min())
heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
print(heightmap.mean())
print(heightmap)

heightmap = np.array(heightmap * 255, np.uint8)

heightmap_image = Image.fromarray(heightmap)
heightmap_image.save('data/heightmap3.png')

store.close()

#exit()
with TiffFile('data/example.tif') as tif:
    for page in tif.pages:
        for tag in page.tags:
            print(tag.name, tag.value)
