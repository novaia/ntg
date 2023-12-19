FROM nvcr.io/nvidia/jax:23.10-py3
RUN python -m pip install -U "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python -m pip install nvidia-dali-cuda120
RUN python -m pip install pillow==10.0.1 pytest==7.4.3 wandb==0.16.0
WORKDIR /project