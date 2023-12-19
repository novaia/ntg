FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN mkdir /checkpoints

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install python3.9 -y
RUN apt install python3-pip -y

WORKDIR /project
COPY requirements.txt requirements.txt

RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3.9 -m pip install -r requirements.txt