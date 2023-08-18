#!/usr/bin/env python
# coding: utf-8

import os

# Modify this according to your Eigen installation and source code
eigen_include_path = os.path.expanduser("~/eigen-3.4.0")
matplotlib_include_path = os.path.expanduser("~")
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
        "-I/usr/include/python3.8",
        "-I/usr/local/lib/python3.8/dist-packages/numpy/core/include"  # Add this line
    ],
    LIBS=["python3.8"]
)

# VariantDir, Program definition, and other settings
env.VariantDir(output_dir, source_dir, duplicate=0)
files = Glob(os.path.join(source_dir, "*.cpp"))
env.Program(os.path.join(output_dir, 'engine.exe'), files)
