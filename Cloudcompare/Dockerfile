FROM ubuntu:22.04

# Устанавливаем переменные окружения для избежания вопросов при установке пакетов
ENV DEBIAN_FRONTEND=noninteractive

# Обновляем список пакетов и устанавливаем обновления
RUN apt-get update && apt-get upgrade -y

# Устанавливаем базовые утилиты и sudo
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


RUN git clone --recursive https://github.com/cloudcompare/CloudCompare.git

RUN apt-get update

RUN apt-get install -y \
    qt6-tools-dev \
    libqt5opengl5-dev \
    libqt5svg5-dev \
    qttools5-dev  \
    qtbase5-dev \
    libpcl-dev

WORKDIR /app/CloudCompare/build

RUN cmake .. \
    -DPLUGIN_STANDARD_QPCL=ON \
    -DPLUGIN_STANDARD_QCSF=ON \
    -DCMAKE_BUILD_TYPE=Release

RUN make -j8
RUN make install

# Переключаемся на нового пользователя (опционально)
USER newuser

CMD CloudCompare

