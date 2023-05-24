# Neural Terrain Generation
Neural Terrain Generation (NTG) is a collection of generative neural networks that output heightmaps for 3D terrain. This repository contains code for the entire NTG training pipeline.

[NTG Unity Trailer](https://youtu.be/MZakPuXyquk)

<img src="./images/display_heightmaps.png" width="800px"></img>

## Setup
After cloning, create a directory at the root of the repository called ``data``.

## Legacy Code
 The ``legacy`` directory contains all the NTG code that was originally implemented in Tensorflow. This code is no longer maintained, but is kept here for reference. In order to run it, use the Docker environment described by the ``legacy/Dockerfile`` and ``legacy/docker-compose.yaml`` files.

## Docker Environment

Building image:
```
docker-compose build
```

Starting container/environment:
```
docker-compose up -d
```

Opening a shell in container:
```
docker-compose exec ntg bash
```

Instead of opening a shell, you can also go to http://localhost:8888/ to access a Jupyter Lab instance running inside the container.

Stopping container/environment:
```
docker-compose down
```