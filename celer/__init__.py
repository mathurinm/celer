"""Celer algorithm to solve the Lasso"""

from .homotopy import celer_path
from .dropin_sklearn import Lasso, LassoCV, LogisticRegression
from .dropin_sklearn import MultiTaskLassoCV, MultiTaskLasso

__version__ = '0.5dev'
