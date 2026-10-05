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

# URDF import (needs tinyxml2).
find_library(TINYXML2_LIBRARY tinyxml2)
find_path(TINYXML2_INCLUDE_DIR tinyxml2.h)
if(TINYXML2_LIBRARY AND TINYXML2_INCLUDE_DIR)
    target_sources(phys PRIVATE src/phys/urdf.cpp)
    target_include_directories(phys PRIVATE ${TINYXML2_INCLUDE_DIR})
    target_link_libraries(phys PUBLIC ${TINYXML2_LIBRARY})
    target_compile_definitions(phys PUBLIC PHYS_HAS_URDF)
else()
    message(STATUS "tinyxml2 not found: building without URDF import")
endif()

# Motion planner (git submodule extern/MotionPlanning, header-only core:
# URDF kinematics, IK, MoveJ / MoveL trajectory optimisation; needs Eigen).
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extern/MotionPlanning/src/robot.hpp AND TARGET Eigen3::Eigen
   AND TINYXML2_LIBRARY)
    add_library(motion_planner INTERFACE)
    target_include_directories(motion_planner INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/MotionPlanning/src ${TINYXML2_INCLUDE_DIR})
    target_link_libraries(motion_planner INTERFACE Eigen3::Eigen ${TINYXML2_LIBRARY})
    target_compile_features(motion_planner INTERFACE cxx_std_17)
else()
    message(STATUS "motion planner unavailable (git submodule update --init; needs Eigen, tinyxml2)")
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
        if(TARGET motion_planner)
            target_link_libraries(_libphys PRIVATE motion_planner)
            target_compile_definitions(_libphys PRIVATE PHYS_HAS_PLANNER)
        endif()
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
    if(TINYXML2_LIBRARY)
        add_executable(test_urdf tests/test_urdf.cpp)
        target_link_libraries(test_urdf PRIVATE phys)
        target_compile_definitions(test_urdf PRIVATE PHYS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        if(TARGET motion_planner)
            target_link_libraries(test_urdf PRIVATE motion_planner)
            target_compile_definitions(test_urdf PRIVATE PHYS_HAS_PLANNER)
        endif()
        add_test(NAME test_urdf COMMAND test_urdf)
    endif()
    add_executable(test_tactile tests/test_tactile.cpp)
    target_link_libraries(test_tactile PRIVATE phys)
    add_test(NAME test_tactile COMMAND test_tactile)

    add_executable(test_tactile_sensor tests/test_tactile_sensor.cpp)
    target_link_libraries(test_tactile_sensor PRIVATE phys)
    add_test(NAME test_tactile_sensor COMMAND test_tactile_sensor)
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
    if(TARGET motion_planner)
        # UR5e: plan with the motion planner, track the plans in libphys
        add_executable(ur5e_planner examples/ur5e_planner.cpp)
        target_link_libraries(ur5e_planner PRIVATE phys motion_planner)
        target_compile_definitions(ur5e_planner PRIVATE PHYS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()
