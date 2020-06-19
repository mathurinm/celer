celer
=====

|image0| |image1|

Fast algorithm to solve Lasso-like problems with dual extrapolation. The package can handle the following problems: Lasso, Sparse Logistic regression, GroupLasso and MultiTaskLasso.

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

Install and work with the development version
=============================================

From a console or terminal clone the repository and install Celer:

::

    git clone https://github.com/mathurinm/celer.git
    cd celer/
    pip install -e .

To build the documentation you will need to run:


::

    pip install -U sphinx_gallery sphinx_bootstrap_theme
    cd doc/
    make html


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

All dependencies are in ``./setup.py`` file.

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

    @article{massias2019dual,
    title={Dual Extrapolation for Sparse Generalized Linear Models},
    author={Massias, Mathurin and Vaiter, Samuel and Gramfort, Alexandre and Salmon, Joseph},
    journal={arXiv preprint arXiv:1907.05830},
    year={2019}
    }


ArXiv links:

- https://arxiv.org/abs/1802.07481
- https://arxiv.org/abs/1907.05830

.. |image0| image:: https://travis-ci.com/mathurinm/celer.svg?branch=master
   :target: https://travis-ci.com/mathurinm/celer/
.. |image1| image:: https://codecov.io/gh/mathurinm/celer/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mathurinm/celer
