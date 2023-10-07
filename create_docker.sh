#!/bin/bash

# Check if image exists
IMAGE_NAME="geocosicorr3d_conda_py37:latest"
TAR_PATH="/home/saif/docker_images/geocosicorr3d_conda_py37.tar"

if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    # Image doesn't exist, so load it from tar
    docker load -i $TAR_PATH
fi

# Build your image
docker build -t geocosicorr3d:latest .
