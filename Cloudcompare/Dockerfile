FROM ubuntu:22.04

# Устанавливаем переменные окружения для избежания вопросов при установке пакетов
ENV DEBIAN_FRONTEND=noninteractive

# Обновляем список пакетов и устанавливаем обновления
RUN apt-get update && apt-get upgrade -y

# Устанавливаем базовые утилиты и sudo
RUN apt-get install -y \
    curl \
    wget \
    tar \
    git \
    vim \
    gcc \
    g++ \
    make \
    gedit \
    net-tools \
    iputils-ping \
    software-properties-common \
    build-essential \
    libssl-dev \
    sudo

RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev

RUN pip install numpy
RUN pip install open3d
RUN pip install catboost imblearn
RUN pip install pypcd4

RUN pip install matplotlib
RUN pip install tensorflow
RUN pip install scikit-image

# устанавливаем CMAKE
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2.tar.gz
RUN tar xzf cmake-3.24.2.tar.gz
WORKDIR cmake-3.24.2
RUN ./configure
RUN make
RUN make install


# (Опционально) Устанавливаем локаль
RUN apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Создаем пользователя с домашней директорией и добавляем в sudo
RUN useradd -m -s /bin/bash newuser && \
    echo 'newuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Очищаем кеш apt для уменьшения размера образа
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию по умолчанию
WORKDIR /app


RUN git clone -b v2.12.4 --recursive https://github.com/cloudcompare/CloudCompare.git

RUN apt-get update

RUN apt-get install -y \
    qt6-tools-dev \
    libqt5opengl5-dev \
    libqt5svg5-dev \
    qttools5-dev  \
    qtbase5-dev \
    libpcl-dev \
    libqt6svg6-dev


WORKDIR /app/CloudCompare/plugins/core/Standard
COPY ./qCirno ./qCirno/
RUN sed -i '18i\add_subdirectory( qCirno )' CMakeLists.txt


WORKDIR /app/CloudCompare/build

RUN cmake .. \
    -DPLUGIN_STANDARD_QPCL=ON \
    -DPLUGIN_STANDARD_QCSF=ON \
    -DPLUGIN_STANDARD_QCIRNO=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-deprecated-copy" \
    -DCOMPILE_OPTIONS_FOR_WARNINGS_AS_ERRORS=OFF

RUN make -j8
RUN make install

# Переключаемся на нового пользователя (опционально)
USER newuser

#CMD ["sudo", "CloudCompare"]


