#!/bin/bash

docker run --rm \
 -e DISPLAY=$DISPLAY \
 -e OPEN3D_DIR=/lib/open3d \
 -e LD_LIBRARY_PATH=/lib/open3d/lib \
 -e CPLUS_INCLUDE_PATH=/lib/open3d/include \
 -e CMAKE_PREFIX_PATH=/lib/open3d/lib/cmake/Open3D \
 --mount type=bind,src=/dev/,target=/dev/,consistency=cached \
 --mount type=bind,src=/dev/dri,target=/dev/dri,consistency=cached \
 --mount type=bind,src=/tmp/.X11-unix,target=/tmp/.X11-unix,consistency=cached \
 --mount type=bind,src=/home/ivanskurikhinadmin,target=/workspace/Server \
 --name ubuntuhumble \
 --gpus all \
 -it ubuntu22:latest


