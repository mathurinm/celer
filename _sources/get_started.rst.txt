===========
Get started
===========

In this starter examples, we will fit a Lasso estimator on a toy dataset. 

Beforehand, make sure to install ``celer``::

    $ pip install -U celer



Generate a toy dataset
----------------------

``celer`` comes with a module, :ref:`Datasets fetchers`,
that expose several functions to fetch/generate datasets.
We are going to use ``make_correlated_data`` to generate a toy dataset.

.. code-block:: python

    # imports
    from celer.datasets import make_correlated_data
    from sklearn.model_selection import train_test_split

    # generate the toy dataset
    X, y, _ = make_correlated_data(n_samples=500, n_features=5000)
    # split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)



Fit and score a Lasso estimator
-------------------------------

``celer`` exposes easy-to-use to use estimators as it was designed under the ``scikit-learn``
API. ``celer`` also integrates well with it (e.g. the ``Pipeline`` and ``GridSearchCV``).


.. code-block:: python

    # import model
    from celer import Lasso

    # init and fit
    model = Lasso()
    model.fit(X_train, y_train)

    # print R²
    print(model.score(X_test, y_test))



Perform cross-validation
------------------------

``celer`` Lasso estimator comes with native cross-validation. 
The following snippets performs cross-validation on a grid 100 ``alphas`` using 5 folds. 
And look how fast ``celer`` is compared to the ``scikit-learn``.

.. code-block:: python

    # imports
    import time
    from celer import LassoCV
    from sklearn.linear_model import LassoCV as sk_LassoCV

    # fit for celer
    start = time.time()
    celer_lassoCV = LassoCV(n_alphas=100, cv=5)
    celer_lassoCV.fit(X, y)
    print(f"time elapsed for celer LassoCV: {time.time() - start}")

    # fit for scikit-learn
    start = time.time()
    sk_lassoCV = sk_LassoCV(n_alphas=100, cv=5)
    sk_lassoCV.fit(X, y)
    print(f"time elapsed for scikit-learn LassoCV: {time.time() - start}")


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    time elapsed for celer LassoCV: 5.062559127807617
    time elapsed for scikit-learn LassoCV: 27.427260398864746



Further links
-------------

This was just a starter example. Get familiar with ``celer`` by browsing its :ref:`API documentation` or
explore the :ref:`Examples Gallery`, which includes examples on real-life datasets as well as 
timing comparison with other solvers.