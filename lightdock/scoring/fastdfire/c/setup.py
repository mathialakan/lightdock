from distutils.core import setup, Extension
import numpy as np
'''
setup(ext_modules=[Extension("cdfire",
                     ["cdfire.c"],
                     extra_compile_args=[ "-noswitcherror", "-acc", "-O3", "-fPIC"],
                     extra_link_args=["-noswitcherror", "-shared", "-acc","-lm"],
                     include_dirs = [np.get_include()])])

'''

setup(
    ext_modules=[Extension("cdfire", ["cdfire.c"],
                 extra_compile_args=[ "-fopenacc", "-foffload=nvptx-none", "-O3"],
                 extra_link_args=["-shared", "-fopenacc", "-foffload=nvptx-none", "-lm", "-lgomp",
                                  "-Wl,-rpath=/sw/summit/gcc/9.1.0-alpha+20190716/lib64"],
                 include_dirs = [np.get_include()])]
)
