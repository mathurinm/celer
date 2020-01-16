"""Celer algorithm to solve the Lasso"""

from .homotopy import celer_path
from .wrapper import celer
from .dropin_sklearn import Lasso, LassoCV, LogisticRegression

__version__ = '0.5dev'
