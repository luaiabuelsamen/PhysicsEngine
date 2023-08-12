env = Environment(CXX="g++", CXXFLAGS=["-std=c++14"])


# VariantDir, Program definition, and other settings
env.VariantDir('exe', 'source', duplicate=0)
files = Glob('./source/*.cpp')
env.Program('exe/engine.exe', files)
