.. celer documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Celer
======

This is a library to run the Constraint Elimination for the Lasso with Extrapolated Residuals (Celer) algorithm [1].
Currently, the package handles the following problems:

- Lasso
- weighted Lasso
- Sparse Logistic regression
- Group Lasso
- Multitask Lasso.

The estimators follow the scikit-learn API, come with automated parallel cross-validation, and support both sparse and dense data, with optionally feature centering, normalization, and unpenalized intercept fitting.
The solvers used allow for solving large scale problems with millions of features, up to 100 times faster than scikit-learn.


Install the released version
----------------------------

From a console or terminal install celer with pip:

::

    pip install -U celer


Install the development version
-------------------------------

We recommend to use the `Anaconda Python distribution <https://www.continuum.io/downloads>`_.
First clone the repository available at https://github.com/mathurinm/celer::

    $ git clone https://github.com/mathurinm/celer.git
    $ cd celer/
    $ pip install -e .

To check if everything worked fine, you can do::

    $ python -c 'import celer'

and it should not give any error message.

From a Python shell you can just do::

    >>> import celer


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

    @article{massias2020dual,
    author  = {Mathurin Massias and Samuel Vaiter and Alexandre Gramfort and Joseph Salmon},
    title   = {Dual Extrapolation for Sparse GLMs},
    journal = {Journal of Machine Learning Research},
    year    = {2020},
    volume  = {21},
    number  = {234},
    pages   = {1-33},
    url     = {http://jmlr.org/papers/v21/19-587.html}
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
