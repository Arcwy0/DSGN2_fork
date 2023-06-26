#!/bin/sh

# Build script for Docker image targeting Jetson AGX Xavier

# Set the Dockerfile and image tag
# DOCKERFILE="Dockerfile.aarch64"
IMAGE_TAG="dsgn2/arm64:dsgn2"

cd "$(dirname "$0")"
cd ..
export PRODUCTION_PATH=$PWD
export ARCH=`uname -m`
export NUM_THREADS=`nproc`

docker-compose --env-file $PRODUCTION_PATH/build.env \
    -f $PRODUCTION_PATH/build.yml \
    up --build $@

# Save the Docker image to a tar file
# docker save $IMAGE_TAG -o $IMAGE_TAG.tar
