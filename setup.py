from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(name='celer',
      version='0.1',
      description='Fast WS algorithm with dual extrapolation for the Lasso',
      author='Mathurin Massias',
      author_email='mathurin.massias@gmail.com',
      url='',
      packages=['celer'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('celer.sparse',
                    sources=['celer/sparse.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
          Extension('celer.dense',
                    sources=['celer/dense.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
          Extension('celer.utils',
                    sources=['celer/utils.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
                 ],
      )
