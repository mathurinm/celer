from sklearn.linear_model import (Lasso as Lasso_sklearn,
                                  LassoCV as LassoCV_sklearn)

from .homotopy import celer_path


class Lasso(Lasso_sklearn):
    """Lasso scikit-learn estimator based on CELER solver

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    max_iter : int, optional
        The maximum number of iterations

    gap_freq : int
        XXX

    max_epochs : int
        XXX

    p0 : int
        XXX

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    verbose : bool
        XXX

    prune : bool
        XXX

    XXX : why no fit_intercept?

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : int | array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from celer import Lasso
    >>> clf = Lasso(alpha=0.1)
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    >>> print(clf.coef_)
    [0.85 0.  ]
    >>> print(clf.intercept_)  # doctest: +ELLIPSIS
    0.15...

    See also
    --------
    celer_path
    LassoCV

    Notes
    -----
    XXX : add ref to CELER paper
    """

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
    """LassoCV scikit-learn estimator based on CELER solver"""

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
