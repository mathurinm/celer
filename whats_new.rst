Version 0.6.1
-----------

Changelog
~~~~~~~~~
- Rely on the libsvmdata package to donwload datasets from LIBSVM.


Version 0.6
-----------

Changelog
~~~~~~~~~
- Added `weights` to the Lasso estimator.
- Added `make_correlated_data` to the `datasets` module, to generate simulations with Toeplitz correlated design.


Version 0.5
-----------

Changelog
~~~~~~~~~
- Structure of `~/celer_data/` folder changed, consider deleting it and redownloading the datasets.
- Added module `datasets`, supporting more datasets (climate for Sparse Group Lasso)
- Removed `celer_logreg` function, use `celer` instead with `pb="logreg"`
- Added sklearn-like `LogisticRegression` class.

Version 0.4
-----------

Changelog
~~~~~~~~~
- Faster homotopy by precomputing norms_X_col and passing residuals from one alpha to the next.


Version 0.3.1
-------------

Changelog
~~~~~~~~~
- Fixed bugs in screening.
