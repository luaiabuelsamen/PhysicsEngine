# CUDA SCons Tool integration for PhysicsEngine
# This file is included by SConstruct to add CUDA support

def exists(env):
    return env.Detect('nvcc')

def generate(env):
    from SCons.Script import Builder, Action
    if not env.Detect('nvcc'):
        print('Warning: nvcc (CUDA compiler) not found! CUDA files may not compile.')
    env['NVCC'] = '/usr/local/cuda-12.6/bin/nvcc'
    env['CUDACOM'] = '$NVCC -o $TARGET -c $SOURCE $NVCCFLAGS $CPPFLAGS $CPPDEFINES $_CPPINCFLAGS'
    env['BUILDERS']['CudaObject'] = Builder(
        action=Action('$CUDACOM', '$CUDACOMSTR'),
        suffix='.o',
        src_suffix='.cu'
    )
    if 'NVCCFLAGS' not in env:
        env['NVCCFLAGS'] = ['-arch=sm_52', '--compiler-options', "'-fPIC'"]
