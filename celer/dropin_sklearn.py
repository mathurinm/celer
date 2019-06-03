# flake8: noqa F401
import inspect
import numpy as np

from scipy import sparse
from abc import ABCMeta, abstractmethod
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_array
from sklearn.externals import six
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import check_cv
from sklearn.externals.joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.linear_model import ElasticNetCV, lasso_path
from sklearn.linear_model import (Lasso as Lasso_sklearn,
                                  LassoCV as _LassoCV)
from sklearn.linear_model.coordinate_descent import (LinearModelCV as
                                                     _LinearModelCV)
from sklearn.linear_model.coordinate_descent import (_alpha_grid,
                                                     _path_residuals)

from .homotopy import celer_path

# Hack because `model = Lasso()` is hardcoded in _LinearModelCV definition
lines = inspect.getsource(_LinearModelCV)
exec(lines)  # when this is executed Lasso is our class, not sklearn's
lines = inspect.getsource(_LassoCV)
lines = lines.replace('LassoCV', 'LassoCV_sklearn')
exec(lines)


class Lasso(Lasso_sklearn):
    """Lasso scikit-learn estimator based on Celer solver

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * ||w||_1

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square.
        For numerical reasons, using ``alpha = 0`` with the
        ``Lasso`` object is not advised.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    gap_freq : int
        Number of coordinate descent epochs between each duality gap
        computations.

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    Examples
    --------
    >>> from celer import Lasso
    >>> clf = Lasso(alpha=0.1)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    Lasso(alpha=0.1, gap_freq=10, max_epochs=50000, max_iter=100,
    p0=10, prune=0, tol=1e-06, verbose=0)
    >>> print(clf.coef_)
    [0.85 0.  ]
    >>> print(clf.intercept_)
    0.15

    See also
    --------
    celer_path
    LassoCV

    References
    ----------
    .. [1] M. Massias, A. Gramfort, J. Salmon
       "Celer: a Fast Solver for the Lasso wit Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html
    """

    def __init__(self, alpha=1., max_iter=100, gap_freq=10,
                 max_epochs=50000, p0=10, verbose=0, tol=1e-4, prune=0,
                 fit_intercept=True, normalize=False, warm_start=False,
                 positive=False):
        super(Lasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.positive = positive

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute Lasso path with Celer."""
        results = celer_path(
            X, y, alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            gap_freq=self.gap_freq, max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune,
            positive=self.positive,
            X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))

        return results


class LassoCV(LassoCV_sklearn):
    """LassoCV scikit-learn estimator based on Celer solver

    The best model is selected by cross-validation.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * ||w||_1

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, sklearn `KFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    gap_freq : int, optional (default=10)
        In the inner loop, the duality gap is computed every `gap_freq`
        coordinate descent epochs.

    max_epochs : int, optional (default=50000)
        Maximum number of coordinate descent epochs when solving a subproblem.

    p0 : int, optional (default=10)
        Number of features in the first working set.

    prune : bool, optional (default=False)
        Whether to use pruning when growing the working sets.

    precompute : ignored parameter, kept for sklearn compatibility.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    intercept_ : float
        independent term in decision function.

    mse_path_ : array, shape (n_alphas, n_folds)
        mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,)
        The grid of alphas used for fitting

    dual_gap_ : ndarray, shape ()
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    See also
    --------
    celer_path
    Lasso
    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, max_iter=100,
                 tol=1e-4, cv=None, verbose=0, gap_freq=10,
                 max_epochs=50000, p0=10, prune=0, precompute='auto',
                 positive=False):
        super(LassoCV, self).__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=tol, cv=cv, fit_intercept=fit_intercept, normalize=normalize,
            verbose=verbose)
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.positive = positive

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path with Celer."""
        alphas, coefs, dual_gaps = celer_path(
            X, y, alphas=alphas, coef_init=coef_init, max_iter=self.max_iter,
            gap_freq=self.gap_freq, max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune,
            positive=self.positive,
            X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))
        return (alphas, coefs, dual_gaps)
