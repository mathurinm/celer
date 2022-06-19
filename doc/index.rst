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



Cite
----

``celer`` is an open source package licensed under 
the `BSD 3-Clause License <https://github.com/mathurinm/celer/blob/main/LICENSE>`_.
Hence, you are free to use it. And if you do so, do not forget to cite:


.. code-block:: bibtex

    @InProceedings{pmlr-v80-massias18a,
      title     = {Celer: a Fast Solver for the Lasso with Dual Extrapolation},
      author    = {Massias, Mathurin and Gramfort, Alexandre and Salmon, Joseph},
      booktitle = {Proceedings of the 35th International Conference on Machine Learning},
      pages     = {3321--3330},
      year      = {2018},
      volume    = {80},
    }


.. code-block:: bibtex

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
   
``celer`` is a outcome of perseverant research. Here are the links to the original papers: 

- `Celer: a Fast Solver for the Lasso with Dual Extrapolation <https://arxiv.org/abs/1802.07481>`_
- `Dual Extrapolation for Sparse GLMs <https://arxiv.org/abs/1907.05830>`_



Explore the documentation
-------------------------

.. toctree::
    :maxdepth: 1

    get_started.rst
    api.rst
    contribute.rst
    auto_examples/index.rst
