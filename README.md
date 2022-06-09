# celer

![build](https://github.com/mathurinm/celer/workflows/build/badge.svg)
![coverage](https://codecov.io/gh/mathurinm/celer/branch/main/graphs/badge.svg?branch=main)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Downloads](https://pepy.tech/badge/celer/month)
![PyPI version](https://badge.fury.io/py/celer.svg)


``celer`` is a Python package that solves Lasso-like problems and provides estimators that follow the ``scikit-learn`` API. Thanks to a tailored implementation, ``celer`` provides a fast solver that tackles large-scale datasets with millions of features **up to 100 times faster than ``scikit-learn``**.

Currently, the package handles the following problems:


| Problem                       | Support Weights | Native cross-validation
| -----------                   | -----------     |----------------
| Lasso                         | yes             | yes
| ElasticNet                    | yes             | yes
| Group Lasso                   | yes             | yes
| Multitask Lasso               | no              | yes
| Sparse Logistic regression    | no              | no



## Why ``celer``?

``celer`` is specially designed to handle Lasso-like problems which makes it a fast solver of such problems.
``celer`` comes particularly with

- automated parallel cross-validation
- support of sparse and dense data
- optional feature centering and normalization
- unpenalized intercept fitting

``celer`` also provides easy-to-use estimators as it is designed under the ``scikit-learn`` API.



## Get started

To get stared, install ``celer`` via pip

```shell
pip install -U celer
```

On your python console,
run the following commands to fit a Lasso estimator on a toy dataset.

```python
>>> from celer Lasso
>>> from celer.datasets import make_correlated_data
>>> X, y, _ = make_correlated_data(n_samples=100, n_features=1000)
>>> estimator = Lasso()
>>> estimator.fit(X, y)
```

This is just a starter examples.
Make sure to browse [``celer`` documentation ](https://mathurinm.github.io/celer/) to learn more about its features.
To get familiar with [``celer`` API](https://mathurinm.github.io/celer/api.html), you can also explore the gallery of examples
which includes examples on real-life datasets as well as timing comparison with other solvers.



## Contribute to celer

``celer`` is an open source project and hence rely on community efforts to evolve.
Your contribution is highly valuable and can come in three forms

- bug report: you may encounter a bug while using ``celer``. Don't hesitate to report it on the [issue section](https://github.com/mathurinm/celer/issues).
- feature request: you may want to extend/add new features to ``celer``. You can use the [issue section](https://github.com/mathurinm/celer/issues) to make suggestions.
- pull request: you may have fixed a bug, enhanced the documentation, ... you can submit a [pull request](https://github.com/mathurinm/celer/pulls) and we will reach out to you asap.

For the last mean of contribution, here are the steps to help you setup ``celer`` on your local machine:

1. Fork the repository and afterwards run the following command to clone it on your local machine

```shell
git clone https://github.com/{YOUR_GITHUB_USERNAME}/celer.git
```

2. ``cd`` to ``celer`` directory and install it in edit mode by running

```shell
cd celer
pip install -e .
```

3. To run the gallery examples and build the documentation, run the followings

```shell
cd doc
pip install -r doc-requirements.txt
make html
```


## Cite

``celer`` is licensed under the [BSD 3-Clause](https://github.com/mathurinm/celer/blob/main/LICENSE). Hence, you are free to use it.
If you do so, please cite:


```tex
    @InProceedings{pmlr-v80-massias18a,
      title     = {Celer: a Fast Solver for the Lasso with Dual Extrapolation},
      author    = {Massias, Mathurin and Gramfort, Alexandre and Salmon, Joseph},
      booktitle = {Proceedings of the 35th International Conference on Machine Learning},
      pages     = {3321--3330},
      year      = {2018},
      volume    = {80},
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
```


## Further links

- https://mathurinm.github.io/celer/
- https://arxiv.org/abs/1802.07481
- https://arxiv.org/abs/1907.05830