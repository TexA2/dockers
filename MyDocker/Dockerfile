FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install wget -y
RUN apt-get install python3 -y
RUN apt-get install python3-dev -y
RUN apt-get install python3-pip -y
RUN pip install pybind11
RUN pip install open3d
RUN pip install open3d-cpu
RUN pip install numpy


WORKDIR /workspace

RUN wget https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz \
    && mkdir -p /lib/open3d \
    && tar -xf open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz -C /lib/open3d --strip-components=1


RUN apt-get install libc++-dev libc++abi-dev -y


#Расскоментить если перестала рабоать графика тогда к sudo нужно будет добавить -E при cmake и make open3d пакетов
#RUN useradd -m -s /bin/bash newuser && \
#    echo 'newuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER newuser
