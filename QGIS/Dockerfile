FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
    curl \
    wget \
    git \
    vim \
    gcc \
    g++ \
    make \
    cmake \
    gedit \
    net-tools \
    iputils-ping \
    software-properties-common \
    sudo

RUN apt-get install -y \
    bison \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    cmake-curses-gui \
    dh-python \
    doxygen \
    expect \
    flex \
    flip \
    gdal-bin \
    git \
    graphviz \
    grass-dev \
    libdraco-dev \
    libexiv2-dev \
    libexpat1-dev \
    libfcgi-dev \
    libgdal-dev \
    libgeos-dev \
    libgsl-dev \
    libpdal-dev \
    libpq-dev \
    libproj-dev \
    libprotobuf-dev \
    libqca-qt5-2-dev \
    libqca-qt5-2-plugins \
    libqscintilla2-qt5-dev \
    libqt5opengl5-dev \
    libqt5serialport5-dev \
    libqt5sql5-sqlite \
    libqt5svg5-dev \
    libqt5webkit5-dev \
    libqt5xmlpatterns5-dev \
    libqwt-qt5-dev \
    libspatialindex-dev \
    libspatialite-dev \
    libsqlite3-dev \
    libsqlite3-mod-spatialite \
    libyaml-tiny-perl \
    libzip-dev \
    libzstd-dev \
    lighttpd locales \
    ninja-build \
    ocl-icd-opencl-dev \
    opencl-headers \
    pandoc \
    pdal \
    pkg-config \
    poppler-utils \
    protobuf-compiler \
    pyqt5-dev \
    pyqt5-dev-tools \
    pyqt5.qsci-dev \
    python3-all-dev \
    python3-autopep8 \
    python3-dev \
    python3-gdal \
    python3-jinja2 \
    python3-lxml \
    python3-mock \
    python3-nose2 \
    python3-owslib \
    python3-plotly \
    python3-psycopg2 \
    python3-pygments \
    python3-pyproj \
    python3-pyqt5 \
    python3-pyqt5.qsci \
    python3-pyqt5.qtmultimedia \
    python3-pyqt5.qtpositioning \
    python3-pyqt5.qtserialport \
    python3-pyqt5.qtsql \
    python3-pyqt5.qtsvg \
    python3-pyqt5.qtwebkit \
    python3-pyqtbuild \
    python3-sip \
    python3-termcolor \
    python3-yaml \
    qt3d-assimpsceneimport-plugin \
    qt3d-defaultgeometryloader-plugin \
    qt3d-gltfsceneio-plugin \
    qt3d-scene2d-plugin \
    qt3d5-dev \
    qtbase5-dev \
    qtbase5-private-dev \
    qtkeychain-qt5-dev \
    qtmultimedia5-dev \
    qtpositioning5-dev \
    qttools5-dev \
    qttools5-dev-tools \
    sip-tools \
    spawn-fcgi \
    xauth \
    xfonts-100dpi \
    xfonts-75dpi \
    xfonts-base \
    xfonts-scalable \
    xvfb

RUN useradd -m -s /bin/bash newuser && \
    echo 'newuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


WORKDIR /app

RUN git clone https://github.com/qgis/QGIS.git

WORKDIR /app/QGIS/build

RUN ccache --set-config sloppiness=pch_defines,time_macros


RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release

RUN make -j8
RUN make install


USER newuser

CMD qgis
