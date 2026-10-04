FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Core build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    scons \
    g++ \
    wget \
    curl \
    python3 \
    python3-dev \
    python3-pip \
    libboost-dev \
    libopencv-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Eigen
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar -xzf eigen-3.4.0.tar.gz && \
    mv eigen-3.4.0 ~/eigen-3.4.0 && \
    ln -s ~/eigen-3.4.0 /usr/local/include/eigen3 && \
    rm eigen-3.4.0.tar.gz

# matplotlib-cpp header
RUN mkdir -p /app/external/matplotlib-cpp && \
    wget -qO /app/external/matplotlib-cpp/matplotlibcpp.h \
    "https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h"

RUN pip3 install numpy matplotlib

WORKDIR /app

# Copy source
COPY CMakeLists.txt /app/
COPY SConstruct /app/
COPY source/ /app/source/
COPY tests/ /app/tests/

# Build rigid body benchmark (CUDA)
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_CUDA_ARCHITECTURES="72;87" && \
    make -j$(nproc)

# Build original spring-mass visualization (CPU-only, optional)
# RUN scons

CMD ["./build/benchmark"]
