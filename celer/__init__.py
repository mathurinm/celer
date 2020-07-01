"""Celer algorithm to solve L1-type regularized problems."""

from .homotopy import celer_path
from .dropin_sklearn import (Lasso, LassoCV, LogisticRegression, GroupLasso,
                             GroupLassoCV, MultiTaskLasso, MultiTaskLassoCV,
                             MCP, MCPCV)


__version__ = '0.5dev'
