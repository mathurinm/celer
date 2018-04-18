import numpy as np

from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import LassoCV as LassoCV_sklearn

from .homotopy import celer_path


class Lasso(Lasso_sklearn):
    """Drop-in replacement for sklearn.linear_model.Lasso."""

    def __init__(self, alpha=1., max_iter=100, gap_freq=10,
                 max_epochs=50000, p0=10, verbose=0, tol=1e-6,
                 prune=0, **kwargs):
        super(Lasso, self).__init__(**kwargs)
        self.alpha = alpha
        self.max_iter = max_iter
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.verbose = verbose
        self.tol = tol
        self.prune = prune
        self.fit_intercept = False  # fit intercept by adding a dummy column
        self.return_n_iter = True  # HACK: n_iter does not mean the same for us

    def path(self, X, y, alphas, **kwargs):
        alphas, coefs, dual_gaps = celer_path(
            X, y, alphas=alphas, max_iter=self.max_iter,
            gap_freq=self.gap_freq, max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune)
        return (alphas, coefs, dual_gaps, [1])
