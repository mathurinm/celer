from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension, find_packages

import numpy as np


setup(packages=find_packages(),
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('celer.lasso_fast',
                    sources=['celer/lasso_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.cython_utils',
                    sources=['celer/cython_utils.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.PN_logreg',
                    sources=['celer/PN_logreg.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.multitask_fast',
                    sources=['celer/multitask_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.group_fast',
                    sources=['celer/group_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
],
)
