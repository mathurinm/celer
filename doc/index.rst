.. celer documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Celer
======

This is a library to run the Constraint Elimination for the Lasso with Extrapolated Residuals (Celer) algorithm [1].
This algorithm uses an extrapolated dual point which enables a tight control of optimality and quick feature identification.

Installation
------------
First clone the repository available at https://github.com/mathurinm/celer::

    $ git clone https://github.com/mathurinm/celer.git
    $ cd celer/


We recommend to use the `Anaconda Python distribution <https://www.continuum.io/downloads>`_.

From a working environment, you can compile the Cython code and install the package with::

    $ pip install -e .

To check if everything worked fine, you can do::

    $ python -c 'import celer'

and it should not give any error message.

From a Python shell you can just do::

    >>> import celer

If you don't want to use Anaconda, you should still be able to install using `pip`.

Cite
----

If you use this code, please cite:

.. code-block:: bibtex

  @InProceedings{pmlr-v80-massias18a,
    title = {Celer: a Fast Solver for the Lasso with Dual Extrapolation},
    author = {Massias, Mathurin and Gramfort, Alexandre and Salmon, Joseph},
    booktitle = {Proceedings of the 35th International Conference on Machine Learning},
    pages = {3321--3330},
    year = 2018,
    volume = 80,
  }

  @article{massias2019dual,
    title = {Dual Extrapolation for Sparse Generalized Linear Models},
    author = {Massias, Mathurin and Vaiter, Samuel and Gramfort, Alexandre and Salmon, Joseph},
    journal = {arXiv preprint arXiv:1907.05830},
    year = 2019,
  }


ArXiv links:

- https://arxiv.org/abs/1802.07481
- https://arxiv.org/abs/1907.05830


Build the documentation
-----------------------

To build the documentation you will need to run:


::

    pip install -U sphinx_gallery sphinx_bootstrap_theme
    cd doc
    make html


API
---

.. toctree::
    :maxdepth: 1

    api.rst
