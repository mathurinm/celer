# Copyright (c) 2018-2022, Celer
# All rights reserved.
#
# License: BSD 3 clause

from setuptools.command.build_ext import build_ext
from setuptools import dist, setup, Extension, find_packages
import os
dist.Distribution().fetch_build_eggs(['numpy>=1.12'])
import numpy as np  # noqa


DISTNAME = 'celer'
DESCRIPTION = 'Fast algorithm with dual extrapolation for sparse problems'
MAINTAINER = 'Mathurin Massias'
MAINTAINER_EMAIL = 'mathurin.massias@gmail.com'

with open('README.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()

LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mathurinm/celer.git'

with open(os.path.join('celer', '__init__.py'), 'r') as fid:
    src = fid.read()
    exec(src)
    VERSION = __version__

URL = 'https://mathurinm.github.io/celer'
CYTHON_MODULES = ['lasso_fast', 'cython_utils',
                  'PN_logreg', 'multitask_fast', 'group_fast']

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      install_requires=['numpy>=1.12', 'seaborn>=0.7', 'scipy>=0.18.0',
                        'matplotlib>=2.0.0', 'Cython>=0.26', 'libsvmdata>=0.3',
                        'scikit-learn>=1.0', 'xarray', 'download', 'tqdm'],
      packages=find_packages(),
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension(f'celer.{cython_module}',
                             sources=[f'celer/{cython_module}.pyx'],
                             language='c++',
                             include_dirs=[np.get_include()],
                             extra_compile_args=["-O3"])
                   for cython_module in CYTHON_MODULES],
      )
