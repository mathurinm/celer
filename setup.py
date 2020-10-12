from setuptools.command.build_ext import build_ext
from setuptools import dist, setup, Extension, find_packages
import os
dist.Distribution().fetch_build_eggs(['numpy>=1.12'])
import numpy as np  # noqa

descr = 'Fast algorithm with dual extrapolation for sparse problems'

version = None
with open(os.path.join('celer', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'celer'
DESCRIPTION = descr
MAINTAINER = 'Mathurin Massias'
MAINTAINER_EMAIL = 'mathurin.massias@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mathurinm/celer.git'
VERSION = version
URL = 'https://mathurinm.github.io/celer'

setup(name='celer',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.rst').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      install_requires=['numpy>=1.12', 'seaborn>=0.7', 'scipy>=0.18.0',
                        'matplotlib>=2.0.0', 'Cython>=0.26',
                        'scikit-learn>=0.23', 'xarray', 'download', 'tqdm'],
      packages=find_packages(),
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
