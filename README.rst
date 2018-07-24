celer
=====

|image0| |image1|

Fast algorithm to solve the Lasso with dual extrapolation

Documentation
=============

Please visit https://mathurinm.github.io/celer/ for the latest version
of the documentation.

Install the released version
============================

Assuming you have a working Python environment, e.g. with Anaconda you
can `install celer with pip <https://pypi.python.org/pypi/celer/>`__.

From a console or terminal install celer with pip:

::

    pip install -U celer

To setup a fully functional environment we recommend you download this
`conda
environment <https://raw.githubusercontent.com/mathurinm/celer/master/environment.yml>`__
and install it with:

::

    conda env create --file environment.yml

Install the development version
===============================

From a console or terminal clone the repository and install CELER:

::

    git clone https://github.com/mathurinm/celer.git
    cd celer/
    conda env create --file environment.yml
    source activate celer-env
    pip install --no-deps -e .

Demos & Examples
================

You find on the documentation examples on the `Leukemia
dataset <https://mathurinm.github.io/celer/auto_examples/plot_leukemia_path.html>`__
(comparison with scikit-learn) and on the `Finance/log1p
dataset <https://mathurinm.github.io/celer/auto_examples/plot_finance_path.html>`__
(more significant, but it takes times to download the data, preprocess
it, and compute the path).

Dependencies
============

All dependencies are in ``./environment.yml``

Cite
====

If you use this code, please cite:

::

    @InProceedings{pmlr-v80-massias18a,
      title = 	 {Celer: a Fast Solver for the Lasso with Dual Extrapolation},
      author = 	 {Massias, Mathurin and Gramfort, Alexandre and Salmon, Joseph},
      booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
      pages = 	 {3321--3330},
      year = 	 {2018},
      volume = 	 {80},
    }

ArXiv link: https://arxiv.org/abs/1802.07481

.. |image0| image:: https://travis-ci.org/mathurinm/celer.svg?branch=master
   :target: https://travis-ci.org/mathurinm/celer/
.. |image1| image:: https://codecov.io/gh/mathurinm/celer/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mathurinm/celer
