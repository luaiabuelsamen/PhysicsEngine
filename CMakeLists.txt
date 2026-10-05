cmake_minimum_required(VERSION 3.18)
project(libphys LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Fall back to common Jetson architectures (Xavier, Orin) when not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "72;87")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

option(PHYS_STRICT_FP
       "Disable FMA contraction so the CPU and CUDA backends match bit for bit" ON)
option(PHYS_BUILD_PYTHON "Build the Python module (needs pybind11)" ON)
option(PHYS_BUILD_TESTS "Build the C++ tests" ON)
option(PHYS_BUILD_BENCHMARKS "Build the benchmarks" ON)
option(PHYS_BUILD_EXAMPLES "Build the example programs" ON)

# --- libphys -------------------------------------------------------------------

add_library(phys STATIC
    src/phys/world.cpp
    src/phys/backend_cpu.cpp
    src/phys/backend_cuda.cu
    src/phys/tactile.cpp
)
target_include_directories(phys PUBLIC src)
target_link_libraries(phys PUBLIC CUDA::cudart)
# Position independent so the Python module can link it.
set_target_properties(phys PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(PHYS_STRICT_FP)
    target_compile_options(phys PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_compile_definitions(phys PUBLIC PHYS_STRICT_FP)
endif()

# The tactile patch as a plain C shared library, for ctypes-based tools.
add_library(phys_tactile SHARED src/phys/tactile.cpp src/phys/tactile_c.cpp)
target_include_directories(phys_tactile PRIVATE src)
target_compile_options(phys_tactile PRIVATE -ffp-contract=off)

# --- Python module ---------------------------------------------------------------

if(PHYS_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development)
    if(Python3_FOUND)
        execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                        OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                        RESULT_VARIABLE pybind11_missing ERROR_QUIET)
    endif()
    if(Python3_FOUND AND NOT pybind11_missing)
        find_package(pybind11 CONFIG REQUIRED)
        pybind11_add_module(_libphys python/bindings.cpp)
        target_link_libraries(_libphys PRIVATE phys)
        # Built next to the package sources: add python/ to PYTHONPATH to use it.
        set_target_properties(_libphys PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/libphys)
    else()
        message(WARNING "pybind11 not found (pip install pybind11): skipping the Python module")
    endif()
endif()

# --- tests ------------------------------------------------------------------------

if(PHYS_BUILD_TESTS)
    enable_testing()
    add_executable(test_phys tests/test_phys.cpp)
    target_link_libraries(test_phys PRIVATE phys)
    add_test(NAME test_phys COMMAND test_phys)
    add_executable(test_tactile tests/test_tactile.cpp)
    target_link_libraries(test_tactile PRIVATE phys)
    add_test(NAME test_tactile COMMAND test_tactile)
    if(TARGET _libphys)
        add_test(NAME test_python
                 COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/python/tests)
        # Prepend to any existing PYTHONPATH (e.g. a Jetson torch build), and
        # keep stray pytest plugins from loading.
        set_tests_properties(test_python PROPERTIES ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:$ENV{PYTHONPATH};PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")
    endif()
endif()

# --- benchmarks -------------------------------------------------------------------

if(PHYS_BUILD_BENCHMARKS)
    # Particle solver against hand-written CPU baselines
    add_executable(benchmark
        benchmarks/benchmark.cpp
        benchmarks/cpu_rigid_body.cpp
        benchmarks/cpu_rigid_body_naive.cpp)
    target_link_libraries(benchmark PRIVATE phys)
    # Rigid solver throughput on batched environments (env-steps per second)
    add_executable(bench_envs benchmarks/bench_envs.cpp)
    target_link_libraries(bench_envs PRIVATE phys)
endif()

# --- examples (render GIFs through an ffmpeg pipe) ---------------------------------

if(PHYS_BUILD_EXAMPLES)
    foreach(example rigid_pile particle_bounce particle_pour)
        add_executable(${example} examples/${example}.cpp)
        target_link_libraries(${example} PRIVATE phys)
    endforeach()
endif()
