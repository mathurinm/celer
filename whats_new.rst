Version 0.5
------------

Changelog
~~~~~~~~~
- Added module `datasets`, supporting more datasets (climate for Sparse Group Lasso)
- Removed `celer_logreg` function, use `celer` instead with `pb="logreg"`
- Added sklearn-like `LogisticRegression` class.

Version 0.4
------------

Changelog
~~~~~~~~~
- Faster homotopy by precomputing norms_X_col and passing residuals from one alpha to the next.


Version 0.3.1
------------

Changelog
~~~~~~~~~
- Fixed bugs in screening.
