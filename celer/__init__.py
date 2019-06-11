"""Celer algorithm to solve the Lasso"""

from .homotopy import celer_path
from .wrapper import celer
from .dropin_sklearn import Lasso, LassoCV

__version__ = '0.4'
