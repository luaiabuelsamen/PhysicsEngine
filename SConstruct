#!/usr/bin/env python
# coding: utf-8

import os

# Modify this according to your Eigen installation and source code
eigen_include_path = os.path.expanduser("~/eigen-3.4.0")
matplotlib_include_path = os.path.expanduser("~")
opencv_include_path = "/usr/include/opencv4"  # Update this path for the correct version
opencv_lib_path = "/usr/lib/x86_64-linux-gnu"  # Update this path for the correct version
source_dir = "source"
output_dir = "exe"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# SCons Environment
env = Environment(
    CXX="g++",
    CXXFLAGS=[
        "-std=c++14",
        "-I{}".format(eigen_include_path),
        "-I{}".format(matplotlib_include_path),
        "-I{}".format(opencv_include_path),  # Add this line for OpenCV
        "-I/usr/include/python3.8",
        "-I/usr/local/lib/python3.8/dist-packages/numpy/core/include"
    ],
    LIBPATH=[opencv_lib_path],  # Add this line for OpenCV library path
    LIBS=["opencv_imgproc", "opencv_core", "opencv_highgui",  "opencv_videoio", "python3.8"]  # Add OpenCV libraries
)

# VariantDir, Program definition, and other settings
env.VariantDir(output_dir, source_dir, duplicate=0)
files = Glob(os.path.join(source_dir, "*.cpp"))
env.Program(os.path.join(output_dir, 'engine.exe'), files)
