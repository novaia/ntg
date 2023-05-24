# Data Pipeline

This folder contains scripts required to create datasets for NTG. This file serves to explain the functionality of each of the scripts, outline various datasets created with them, and explain how to recreate the datasets.

## Script Explanations

### scrape_earth_explorer

- Logs in to [Earth Explorer]('https://earthexplorer.usgs.gov/'), and downloads the SRTM 1 arc-second dataset. This is a dataset of approximately 15k high resolution GEOTIFF heightmaps.

### geotiff_to_png

- Converts GEOTIFFs to PNGs with the rasterio library. Primarily used to convert the SRTM 1 arc-second dataset into a more readable form.

### split_heightmaps

- Splits heightmaps into 100 sub-heightmaps. Useful for transforming SRTM 1 arc-second PNG into something more suitable for training an image generation model.

### train_corrupted_heightmap_discriminator

- The SRTM 1 arc-second PNG dataset contains a number of undesirable or otherwise corrupted heightmaps. These include heighmaps with very little terrain visible, heightmaps with padding issues, and heightmaps with only black and white values. This script trains a convolutional NN to discriminate between corrupted and uncorrupted heightmaps. A handpicked discrimination dataset is required for this.

### isolate_uncorrupted_heightmaps

- Uses a NN (trained by the previous script) to identify and isolate uncorrupted heightmaps from a mixed dataset, thereby creating a new uncorrupted dataset.

### blur_images

- Blurs images to create input for style transfer. Style transfer was never implemented so this was never used.

## Datasets

### SRTM 1 arc-second

- This is the base dataset scraped from [Earth Explorer]('https://earthexplorer.usgs.gov/'). It contains approximately 15k highresolution GEOTIFF heightmaps.
- To "recreate", simply run ``scrape_earthexplorer.py``.

### SRTM 1 arc-second PNG

- SRTM 1 arc-second dataset converted into PNGs.
- Use ``geotiff_to_png.py`` on SRTM 1 arc-second to recreate.