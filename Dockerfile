# Builds libphys, its Python module and tests on an x86-64 CUDA host.
# (On a Jetson, build natively or start from an L4T image such as
# nvcr.io/nvidia/l4t-jetpack instead.)
#
#   docker build -t libphys .
#   docker run --gpus all libphys                 # runs the test suite
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake python3 python3-dev python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy scipy pybind11 pytest torch

WORKDIR /app
COPY CMakeLists.txt ./
COPY src/ src/
COPY python/ python/
COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY examples/ examples/
COPY tools/ tools/

# Volta through Ada; adjust for your GPU.
RUN cmake -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;86;89" && cmake --build build -j"$(nproc)"

ENV PYTHONPATH=/app/python
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
