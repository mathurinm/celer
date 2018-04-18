from sklearn.linear_model import (Lasso as Lasso_sklearn,
                                  LassoCV as LassoCV_sklearn)

from .homotopy import celer_path


class Lasso(Lasso_sklearn):
    """Drop-in replacement for sklearn.linear_model.Lasso."""

    def __init__(self, alpha=1., max_iter=100, gap_freq=10,
                 max_epochs=50000, p0=10, verbose=0, tol=1e-6, prune=0):
        super(Lasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter)
        self.verbose = verbose
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.fit_intercept = False
        self.return_n_iter = True

    def path(self, X, y, alphas, **kwargs):
        alphas, coefs, dual_gaps = celer_path(
            X, y, alphas=alphas, max_iter=self.max_iter,
            gap_freq=self.gap_freq, max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune)
        return (alphas, coefs, dual_gaps, [1])


class LassoCV(LassoCV_sklearn):
    """Drop-in replacement for sklearn.linear_model.LassoCV."""

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=False, max_iter=100,
                 tol=1e-6, cv=None, verbose=0, gap_freq=10,
                 max_epochs=50000, p0=10, prune=0,
                 normalize=False, precompute='auto'):
        super(LassoCV, self).__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=tol, cv=cv, fit_intercept=fit_intercept, verbose=verbose)
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.fit_intercept = False
        self.return_n_iter = True

    def path(self, X, y, alphas, **kwargs):
        alphas, coefs, dual_gaps = celer_path(
            X, y, alphas=alphas, max_iter=self.max_iter,
            gap_freq=self.gap_freq, max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune)
        return (alphas, coefs, dual_gaps)
