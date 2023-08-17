#!/usr/bin/env python
# coding: utf-8

import os

# Modify this according to your Eigen installation and source code
eigen_include_path = os.path.expanduser("~/eigen-3.4.0")
source_dir = "source"
output_dir = "exe"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# SCons Environment
env = Environment(CXX="g++", CXXFLAGS=["-std=c++14", "-I{}".format(eigen_include_path)])

# VariantDir, Program definition, and other settings
env.VariantDir(output_dir, source_dir, duplicate=0)
files = Glob(os.path.join(source_dir, "*.cpp"))
env.Program(os.path.join(output_dir, 'engine.exe'), files)
