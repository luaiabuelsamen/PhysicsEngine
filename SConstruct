#!/usr/bin/env python
# coding: utf-8

import os
import shutil
from SCons.Script import *

# Helper function to get pkg-config flags
def pkg_config(*packages):
    cflags = os.popen(f"pkg-config --cflags {' '.join(packages)}").read().strip().split()
    libs = os.popen(f"pkg-config --libs {' '.join(packages)}").read().strip().split()
    return cflags, libs

# Define Qt modules you want to include
qt_modules = [
    "Qt5Core", "Qt5Gui", "Qt5Widgets", "Qt5Network", "Qt5OpenGL", "Qt5PrintSupport",
    "Qt5Sql", "Qt5Test", "Qt5Xml", "Qt5Concurrent", "Qt5DBus"
]

# Get Qt flags from pkg-config for all specified modules
qt_cflags, qt_libs = pkg_config(*qt_modules)

# Modify paths according to your installation and source code
eigen_include_path = os.path.expanduser("~/eigen-3.4.0")
matplotlib_include_path = os.path.expanduser("~")
opencv_include_path = "/usr/include/opencv4"
opencv_lib_path = "/usr/lib/aarch64-linux-gnu"
source_dir = "source"
output_dir = "exe"
build_dir = os.path.join(source_dir, "build")

# Ensure the output and build directories exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(build_dir):
    os.makedirs(build_dir)

# SCons Environment
env = Environment(
    CXX="g++",
    CXXFLAGS=[
        "-std=c++14",
        f"-I{eigen_include_path}",
        f"-I{matplotlib_include_path}",
        f"-I{opencv_include_path}",
        "-I/usr/include/python3.8",
        "-I/usr/local/lib/python3.8/dist-packages/numpy/core/include"
    ] + qt_cflags,  # Append Qt cflags
    LIBPATH=[opencv_lib_path],
    LIBS=["opencv_imgproc", "opencv_core", "opencv_highgui", "opencv_videoio", "python3.8"] + qt_libs  # Append Qt libs
)

# Add Qt moc builder to handle headers with Q_OBJECT
qt_moc = env.WhereIs('moc') or 'moc'

def moc_builder(source, target, env):
    moc_cmd = f"{qt_moc} -o {target[0]} {source[0]}"
    os.system(moc_cmd)
    
moc = Builder(action=moc_builder, suffix='.moc.cpp', src_suffix='.h')
env.Append(BUILDERS={'Moc': moc})

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
            shutil.move(obj_path, os.path.join(build_dir, os.path.basename(obj_path)))
    return 0

# Register the post-build action
env.AlwaysBuild(env.Command(None, program, move_object_files))
