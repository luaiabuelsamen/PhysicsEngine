FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    scons \
    libboost-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && \
apt install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
apt update && \
apt-get install -y \
g++ \
python3.8 \
python3.8-dev

RUN apt install -y python3.8-distutils && \
apt install -y curl &&\
apt install -y wget &&\
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8

RUN apt install -y build-essential && \
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
tar -xzvf eigen-3.4.0.tar.gz && \
mv eigen-3.4.0 ~/eigen-3.4.0 && \
ln -s ~/eigen-3.4.0 /usr/local/include/eigen3 && \
wget -O ~/matplotlibcpp.h "https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h"

# Install required packages including OpenCV
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN python3.8 -m pip install -r /app/requirements.txt
WORKDIR /app


