celer
=====


A fast solver for Lasso-like problems
-------------------------------------

``celer`` is a Python package that solves Lasso-like problems and provides estimators 
that follow the ``scikit-learn`` API. Thanks to a tailored implementation,
``celer`` provides a fast solver that tackles large-scale datasets with millions of features 
**up to 100 times faster than** ``scikit-learn``.

Currently, the package handles the following problems:

.. list-table:: The supported lasso-like problems
   :header-rows: 1

   * - Problem
     - Support of weights
     - Native cross-validation
   * - Lasso
     - ✓
     - ✓
   * - ElasticNet 
     - ✓
     - ✓
   * - Group Lasso 
     - ✓
     - ✓
   * - Multitask Lasso
     - ✕
     - ✓
   * - Sparse Logistic regression
     - ✕
     - ✕


Why ``celer``?
--------------

``celer`` is specially designed to handle Lasso-like problems which enable it to solve them quickly.
``celer`` comes particularly with

- automated parallel cross-validation
- support of sparse and dense data
- optional feature centering and normalization
- unpenalized intercept fitting

``celer`` also provides easy-to-use estimators as it is designed under the ``scikit-learn`` API.


Install ``celer``
-----------------

``celer`` can be easily installed through the Python package manager ``pip``.
To get the laster version of the package, run::

    $ pip install -U celer

Head directly to the :ref:`Get started` page to get a hands-on example of how to use ``celer``.


Explore the documentation
-------------------------

.. toctree::
    :maxdepth: 1

    get_started.rst
    api.rst
    contribute.rst
    auto_examples/index.rst
    about.rst
