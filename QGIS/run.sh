#!/bin/bash

docker run --rm \
 -e DISPLAY=$DISPLAY \
 --mount type=bind,src=/dev/,target=/dev/,consistency=cached \
 --mount type=bind,src=/dev/dri,target=/dev/dri,consistency=cached \
 --mount type=bind,src=/tmp/.X11-unix,target=/tmp/.X11-unix,consistency=cached \
 --mount type=bind,src=./Share,target=/app/Share,consistency=cached \
 --name astraqgis \
 --gpus all \
 -it astraqgis:latest


