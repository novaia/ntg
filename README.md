# Neural Terrain Generation
Neural Terrain Generation (NTG) is a collection of generative neural networks that output heightmaps for 3D terrain. This repository contains code for the entire NTG training pipeline.

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
nix develop --impure
```
