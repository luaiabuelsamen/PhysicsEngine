#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import subprocess
from SCons.Script import *

# Helper function to get pkg-config flags
def pkg_config(*packages):
    try:
        cflags = subprocess.check_output(
            ["pkg-config", "--cflags"] + list(packages), stderr=subprocess.STDOUT
        ).decode("utf-8").strip().split()
        libs = subprocess.check_output(
            ["pkg-config", "--libs"] + list(packages), stderr=subprocess.STDOUT
        ).decode("utf-8").strip().split()
        return cflags, libs
    except subprocess.CalledProcessError as e:
        print(f"Error running pkg-config for packages: {packages}")
        print(e.output.decode("utf-8"))
        return [], []


# Modify paths according to your installation and source code
eigen_include_path = os.path.expanduser("~/eigen-3.4.0")
matplotlib_include_path = os.path.join("external", "matplotlib-cpp")
opencv_include_path = "/usr/include/opencv4"
opencv_lib_path = "/usr/lib/aarch64-linux-gnu"
local_lib = "/usr/local/lib"
source_dir = "source"
output_dir = "exe"
build_dir = os.path.join(source_dir, "build")

# Ensure matplotlibcpp.h exists, download if not
matplotlib_header = os.path.join(matplotlib_include_path, "matplotlibcpp.h")
if not os.path.exists(matplotlib_header):
    os.makedirs(matplotlib_include_path, exist_ok=True)
    import urllib.request
    url = "https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h"
    print(f"Downloading {url} to {matplotlib_header}...")
    urllib.request.urlretrieve(url, matplotlib_header)
    print("Download complete.")

# Ensure the output and build directories exist
for directory in [output_dir, build_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# SCons Environment
env = Environment(
    CXX="g++",
    CXXFLAGS=[
        "-std=c++14",
        f"-I{eigen_include_path}",
        f"-I{matplotlib_include_path}",
        f"-I{opencv_include_path}",
        f"-I/usr/include/GL",
        "-I/usr/include/python3.10",
        "-I/usr/local/lib/python3.10/dist-packages/numpy/core/include"
    ],
    LIBPATH=[local_lib, opencv_lib_path],    
    LIBS=["opencv_imgproc", "opencv_core", "opencv_highgui", "opencv_videoio", "python3.10", "GL", "glfw", "GLU", "glut"]  # Add 'glut' here
)

# Detect headers with Q_OBJECT and run moc on them
def process_headers(env, headers):
    moc_files = []
    for header in headers:
        with open(str(header), 'r') as f:
            content = f.read()
            if 'Q_OBJECT' in content:
                moc_file = env.Moc(header)
                moc_files.append(moc_file)
    return moc_files

# Get source and header files
source_files = Glob(os.path.join(source_dir, "*.cpp"))
header_files = Glob(os.path.join(source_dir, "*.h"))

# Run moc on headers with Q_OBJECT
moc_files = process_headers(env, header_files)

# Compile each source file to an object file
object_files = [env.Object(src) for src in source_files + moc_files]

# Link all object files into the final executable in the output directory
program = env.Program(target=os.path.join(output_dir, 'engine.exe'), source=object_files)

# Post-build action to move .o files to build_dir
def move_object_files(target, source, env):
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    for obj_file in object_files:
        obj_path = str(obj_file[0])
        if os.path.exists(obj_path):
            # Only move if the file exists
            shutil.move(obj_path, os.path.join(build_dir, os.path.basename(obj_path)))
    return 0

# Register the post-build action
env.AlwaysBuild(env.Command(None, program, move_object_files))
