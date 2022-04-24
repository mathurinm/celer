"""Celer algorithm to solve L1-type regularized problems."""

__version__ = '0.7dev'


if __name__ != '__main__':
    from .homotopy import celer_path
    from .dropin_sklearn import (Lasso, LassoCV, LogisticRegression, GroupLasso,
                                 GroupLassoCV, MultiTaskLasso, MultiTaskLassoCV)
