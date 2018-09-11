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


We recommend to use the `Anaconda Python distribution <https://www.continuum.io/downloads>`_,
and create a conda environment with::

    $ conda env create --file environment.yml

Then, you can compile the Cython code and install the package with::

    $ source activate celer-env
    $ pip install --no-deps -e .

To check if everything worked fine, you can do::

    $ source activate celer-env
    $ python -c 'import celer'

and it should not give any error message.

From a Python shell you can just do::

    >>> import celer

If you don't want to use Anaconda, the list of packages you need to install is in the `environment.yml` file.

Cite
----

If you use this code, please cite:

.. code-block:: None

    @InProceedings{pmlr-v80-massias18a,
                   title = {Celer: a Fast Solver for the Lasso with Dual Extrapolation},
                   author = {Massias, Mathurin and Gramfort, Alexandre and Salmon, Joseph},
                   booktitle = {Proceedings of the 35th International Conference on Machine Learning},
                   pages = {3321--3330},
                   year = {2018},
                   volume = {80},
   }


ArXiv link: https://arxiv.org/abs/1802.07481

ICML link: http://proceedings.mlr.press/v80/massias18a.html


API
---

.. toctree::
    :maxdepth: 1

    api.rst
