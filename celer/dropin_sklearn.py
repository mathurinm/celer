import numpy as np

from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import LassoCV as LassoCV_sklearn
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils import check_X_y, check_array

from .wrapper import celer
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
        return (alphas, coefs, dual_gaps, [0])


# class Lasso(Lasso_sklearn):
#     """Drop-in replacement for sklearn.linear_model.Lasso."""
#
#     def __init__(self, alpha=1., max_iter=100, gap_freq=10,
#                  max_epochs=50000, p0=10, verbose=0, tol=1e-6,
#                  prune=0, **kwargs):
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.gap_freq = gap_freq
#         self.max_epochs = max_epochs
#         self.p0 = p0
#         self.verbose = verbose
#         self.tol = tol
#         self.prune = prune
#         super(Lasso, self).__init__(**kwargs)
#         self.fit_intercept = False
#
#     def fit(self, X, y, check_input=True):
#         if check_input:
#             X, y = check_X_y(X, y, accept_sparse='csc',
#                              order='F', dtype=[np.float64, np.float32],
#                              copy=self.copy_X and self.fit_intercept,
#                              multi_output=True, y_numeric=True)
#             y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
#                             ensure_2d=False)
#
#         if self.fit_intercept:
#             X = add_dummy_feature(X, X.shape[1])
#
#         if self.warm_start and hasattr(self, 'coef_'):
#             beta_init = self.coef_
#         else:
#             beta_init = np.zeros(X.shape[1])
#         import ipdb; ipdb.set_trace()
#         coef, _, gaps, _ = celer(
#             X, y, self.alpha, beta_init=None, max_iter=self.max_iter,
#             gap_freq=self.gap_freq, max_epochs=self.max_epochs, p0=self.p0,
#             verbose=self.verbose, tol=self.tol, prune=self.prune)
#         # intercept added as first column:
#         self.coef_ = coef[1:] if self.fit_intercept else coef
#         self.n_iter_ = gaps.shape[0]
#         if self.fit_intercept:
#             self.intercept_ = coef[0, 0] * X.shape[1]
#
#         return self
