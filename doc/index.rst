.. picard documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Celer
======

This is a library to run the Constraint Elimination for the Lasso with Extrapolated Residuals (Celer) algorithm [1].
This algorithm uses an extrapolated dual point which enables a tight cntrol of optimality and quick feature identification.

Installation
------------

We recommend to use the `Anaconda Python distribution <https://www.continuum.io/downloads>`_,
and create a conda environment with::

    $ conda env create --file environment.yml


To check if everything worked fine, you can do::

    $ source activate celer-env
    $ python -c 'import celer'

and it should not give any error message.



Cite
----

   [1] Mathurin Massias, Alexandre Gramfort and Joseph Salmon,
   "Dual Extrapolation for Faster Lasso Solvers",
   ArXiv Preprint, 2018, https://arxiv.org/abs/1802.07481


API
---

.. toctree::
    :maxdepth: 1

    api.rst
