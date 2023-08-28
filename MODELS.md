# Models

## Pix

Description: vanilla implicit diffusion model (DDIM).

Training script: [pix.py](./models/pix.py)

Full list of script parameters:
- ``--start_epoch``: Epoch to start from when resuming training. Starts from scratch if 0. Default: ``0``
- ``--dataset_path``: Path to training dataset. Default: ``../heightmaps/world-heightmaps-01/``
- ``--model_save_path``: Path to directory where model checkpoints are saved. Default: ``data/pix_checkpoints/``
- ``--model_name``: Name of the model. Used for naming checkpoints and training logs. 
- ``--image_save_path``: Path to directory where images generated at the end of each epoch are saved.
- ``--log_file``: Path to log file.
- ``--batch_size``: Batch size for training.
- ``--learning_rate``: Learning rate for Adam optimizer.
- ``--epochs``: Number of epochs to train for.
- ``--image_width``: Width of image to load from dataset.
- ``--image_height``: Height of image to load from dataset.
- ``--use_fid``: If true, use FID to benchmark model at the end of every epoch.
- ``--fid_stats_path``: Path to file containing precomputed FID stats for training dataset.
- ``--fid_batch_size``: Batch size for generating FID samples.
- ``--num_fid_samples``: Number of samples to generate for FID benchmark.
- ``--docker``: If true, the program will expect to be running inside a docker container. Default: ``True``
- ``--export``: 'If true, the model will be exported to a TF SavedModel instead of training. Default: ``False``

## Hyper

WIP
