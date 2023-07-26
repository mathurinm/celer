"""Celer algorithm to solve L1-type regularized problems."""

from .homotopy import celer_path
from .dropin_sklearn import (ElasticNet, ElasticNetCV,
                             GroupLasso, GroupLassoCV,
                             Lasso, LassoCV, LogisticRegression,
                             MultiTaskLasso, MultiTaskLassoCV)


__version__ = '0.7.3'
