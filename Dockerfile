FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y

WORKDIR /project
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]