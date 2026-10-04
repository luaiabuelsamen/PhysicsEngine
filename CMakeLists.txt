cmake_minimum_required(VERSION 3.18)
project(PhysicsEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Fall back to common Jetson architectures (Xavier, Orin) when not specified
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

find_package(CUDAToolkit REQUIRED)

option(PHYS_STRICT_FP
       "Disable FMA contraction so the CPU and CUDA backends match bit for bit" ON)

# libphys: batched rigid body simulation with CPU and CUDA backends
add_library(phys STATIC
    source/phys/world.cpp
    source/phys/backend_cpu.cpp
    source/phys/backend_cuda.cu
)
target_include_directories(phys PUBLIC source)
target_link_libraries(phys PUBLIC CUDA::cudart)
if(PHYS_STRICT_FP)
    target_compile_options(phys PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_compile_definitions(phys PUBLIC PHYS_STRICT_FP)
endif()

# Benchmark: libphys (CPU + CUDA) against hand-written CPU baselines
add_executable(benchmark
    source/benchmark.cpp
    source/cpu_rigid_body.cpp
    source/cpu_rigid_body_naive.cpp
)
target_link_libraries(benchmark PRIVATE phys)

# Rigid solver throughput on batched environments (env-steps per second)
add_executable(bench_envs source/bench_envs.cpp)
target_link_libraries(bench_envs PRIVATE phys)

# Visualizations (render GIFs via an ffmpeg pipe)
add_executable(visualize source/visualize.cpp)
target_link_libraries(visualize PRIVATE phys)

add_executable(visualize_pour source/visualize_pendulum.cpp)
target_link_libraries(visualize_pour PRIVATE phys)

add_executable(visualize_rigid source/visualize_rigid.cpp)
target_link_libraries(visualize_rigid PRIVATE phys)

# Double pendulum chaos visualization (CUDA-parallel RK4)
add_executable(visualize_chaos
    source/visualize_chaos.cpp
    source/cuda_pendulum.cu
)
target_link_libraries(visualize_chaos PRIVATE CUDA::cudart)

# Tests
enable_testing()
add_executable(test_phys tests/test_phys.cpp)
target_link_libraries(test_phys PRIVATE phys)
add_test(NAME test_phys COMMAND test_phys)
