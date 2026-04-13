cmake_minimum_required(VERSION 3.18)
project(PhysicsEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Detect GPU architecture automatically, fall back to common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "72;87")
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Benchmark executable (rigid body engine + benchmark harness)
add_executable(benchmark
    source/benchmark.cpp
    source/cpu_rigid_body.cpp
    source/cpu_rigid_body_naive.cpp
    source/cuda_rigid_body.cu
)

target_include_directories(benchmark PRIVATE source)

# Link CUDA runtime
target_link_libraries(benchmark PRIVATE cudart)

# Enable separable compilation for thrust
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Visualization executable (renders GIF via ffmpeg pipe)
add_executable(visualize
    source/visualize.cpp
    source/cpu_rigid_body.cpp
    source/cuda_rigid_body.cu
)

target_include_directories(visualize PRIVATE source)
target_link_libraries(visualize PRIVATE cudart)
set_target_properties(visualize PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Particle pour visualization (GPU-accelerated)
add_executable(visualize_pour
    source/visualize_pendulum.cpp
    source/cpu_rigid_body.cpp
    source/cuda_rigid_body.cu
)
target_include_directories(visualize_pour PRIVATE source)
target_link_libraries(visualize_pour PRIVATE cudart)
set_target_properties(visualize_pour PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Double pendulum chaos visualization (CUDA-parallel RK4)
add_executable(visualize_chaos
    source/visualize_chaos.cpp
    source/cuda_pendulum.cu
)
target_include_directories(visualize_chaos PRIVATE source)
target_link_libraries(visualize_chaos PRIVATE cudart)
set_target_properties(visualize_chaos PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
