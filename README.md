# Neural Terrain Generation
Neural Terrain Generation (NTG) is a collection of generative neural networks that output heightmaps for 3D terrain. This repository contains code for the entire NTG training pipeline.

## Setup
After cloning, create a directory at the root of the repository called ``data``.

## Computing FID Stats
In order to compute FID stats, you'll need pretrained InceptionV3 weights. You can get these [here](https://huggingface.co/hayden-donnelly/inception-v3-fid/tree/main)
in the form of ``inception_v3_fid.pickle``. Once you've downloaded this file, place it in the ``data`` directory at the root of the repository.

If you would like to track FID as a model trains, you'll need to pre-compute the FID stats for the target dataset.
This can be done by running the following command:
```
python fid --precompute --img_dir <PATH_TO_DATASET> --out_dir <PATH_TO_OUTPUT_DIRECTORY> --img_size <WIDTH> <HEIGHT>
```
Here's a specific example of the command:
```
python fid --precompute --img_dir ../heightmaps/world-heightmaps-01 --out_dir data/dataset_info --img_size 256 256
```
Outside of training, the FID of two datasets can be computed by specifying the path1 and path2 arguments instead of the img_dir argument:
```
python fid --path1 <PATH_TO_DATASET1> --path2 <PATH_TO_DATASET2> --out_dir <PATH_TO_OUTPUT_DIRECTORY> --img_size <WIDTH> <HEIGHT>
```

Full list of parameters:
- ``--path1``: Path to image directory or .npz file containing pre-computed statistics. Default: ``None``
- ``--path2``: Path to image directory or .npz file containing pre-computed statistics. Default: ``None``
- ``--batch_size``: Batch size per device for computing the Inception activations. Default: ``50``
- ``--img_size``: Resize images to this size. The format is (height, width). Default: ``None``, ``None``
- ``--precompute``: If True, pre-compute statistics for given image directory. Default: ``False``
- ``--img_dir``: Path to image directory for pre-computing statistics. Default: ``None``
- ``--out_dir``: Path where pre-computed statistics are stored. Default: ``None``
- ``--out_name``: Name of outputted statistics file. Default: ``stats``
- ``--mmap``: If True, use mmap to compute statistics. Helpful for large datasets. Default: ``True``
- ``--mmap_filename``: Name for mmap file. Only used if mmap is True. Default: ``data/temp/mmap_file``

## Models
- [Terra](./models/terra.py): a purely convolutional diffusion model.

## Development Environment
The NTG development environment is managed with Nix. You can follow the steps below to get started.
1. Install Nix with the [official installer](https://nixos.org/download/) or the [determinate installer](https://github.com/DeterminateSystems/nix-installer).
2. Enable the experimental Nix Flakes feature by adding the following line to ``~/.config/nix/nix.conf`` or ``/etc/nix/nix.conf`` 
(this step can be skipped if you installed nix with the [determinate installer](https://github.com/DeterminateSystems/nix-installer)).
```
experimental-features = nix-command flakes
```
3. Run the following command to open a development shell with all the dependencies installed.
```
nix develop
```
