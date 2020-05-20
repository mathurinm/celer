"""This file is a hack until sklearn=0.24 is released, including a new
inheritance mechanism for LinearModelCV. We will delete it then."""
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse

from sklearn.base import MultiOutputMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import check_cv
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.linear_model._coordinate_descent import (_alpha_grid,
                                                      _path_residuals)
from joblib import Parallel, delayed, effective_n_jobs


class sklearn_LinearModelCV(MultiOutputMixin, LinearModel, metaclass=ABCMeta):
    """Base class for iterative model fitting along a regularization path"""

    @abstractmethod
    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                 normalize=False, precompute='auto', max_iter=1000, tol=1e-4,
                 copy_X=True, cv=None, verbose=False, n_jobs=None,
                 positive=False, random_state=None, selection='cyclic'):
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    @abstractmethod
    def _get_estimator(self):
        """Model to be fitted after the best alpha has been determined."""

    @abstractmethod
    def _is_multitask(self):
        """Bool indicating if class is meant for multidimensional target."""

    def fit(self, X, y):
        """Fit linear model with coordinate descent

        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        """
        # This makes sure that there is no duplication in memory.
        # Dealing right with copy_X is important in the following:
        # Multiple functions touch X and subsamples of X and can induce a
        # lot of duplication of memory
        copy_X = self.copy_X and self.fit_intercept

        check_y_params = dict(copy=False, dtype=[np.float64, np.float32],
                              ensure_2d=False)
        if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
            # Keep a reference to X
            reference_to_old_X = X
            # Let us not impose fortran ordering so far: it is
            # not useful for the cross-validation loop and will be done
            # by the model fitting itself

            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(accept_sparse='csc',
                                  dtype=[np.float64, np.float32], copy=False)
            X, y = self._validate_data(X, y,
                                       validate_separately=(check_X_params,
                                                            check_y_params))
            if sparse.isspmatrix(X):
                if (hasattr(reference_to_old_X, "data") and
                   not np.may_share_memory(reference_to_old_X.data, X.data)):
                    # X is a sparse matrix and has been copied
                    copy_X = False
            elif not np.may_share_memory(reference_to_old_X, X):
                # X has been copied
                copy_X = False
            del reference_to_old_X
        else:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(accept_sparse='csc',
                                  dtype=[np.float64, np.float32], order='F',
                                  copy=copy_X)
            X, y = self._validate_data(X, y,
                                       validate_separately=(check_X_params,
                                                            check_y_params))
            copy_X = False

        if y.shape[0] == 0:
            raise ValueError("y has 0 samples: %r" % y)

        if not self._is_multitask():
            if y.ndim > 1 and y.shape[1] > 1:
                raise ValueError("For multi-task outputs, use "
                                 "MultiTask%s" % self.__class__.__name__)
            y = column_or_1d(y, warn=True)
        else:
            if sparse.isspmatrix(X):
                raise TypeError("X should be dense but a sparse matrix was"
                                "passed")
            elif y.ndim == 1:
                raise ValueError("For mono-task outputs, use "
                                 "%sCV" % self.__class__.__name__[9:])

        model = self._get_estimator()

        if self.selection not in ["random", "cyclic"]:
            raise ValueError("selection should be either random or cyclic.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (X.shape[0], y.shape[0]))

        # All LinearModelCV parameters except 'cv' are acceptable
        path_params = self.get_params()
        if 'l1_ratio' in path_params:
            l1_ratios = np.atleast_1d(path_params['l1_ratio'])
            # For the first path, we need to set l1_ratio
            path_params['l1_ratio'] = l1_ratios[0]
        else:
            l1_ratios = [1, ]
        path_params.pop('cv', None)
        path_params.pop('n_jobs', None)

        alphas = self.alphas
        n_l1_ratio = len(l1_ratios)
        if alphas is None:
            alphas = [_alpha_grid(X, y, l1_ratio=l1_ratio,
                                  fit_intercept=self.fit_intercept,
                                  eps=self.eps, n_alphas=self.n_alphas,
                                  normalize=self.normalize, copy_X=self.copy_X)
                      for l1_ratio in l1_ratios]
        else:
            # Making sure alphas is properly ordered.
            alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratio, 1))
        # We want n_alphas to be the number of alphas used for each l1_ratio.
        n_alphas = len(alphas[0])
        path_params.update({'n_alphas': n_alphas})

        path_params['copy_X'] = copy_X
        # We are not computing in parallel, we can modify X
        # inplace in the folds
        if effective_n_jobs(self.n_jobs) > 1:
            path_params['copy_X'] = False

        # init cross-validation generator
        cv = check_cv(self.cv)

        # Compute path for all folds and compute MSE to get the best alpha
        folds = list(cv.split(X, y))
        best_mse = np.inf

        # We do a double for loop folded in one, in order to be able to
        # iterate in parallel on l1_ratio and folds
        jobs = (delayed(_path_residuals)(X, y, train, test, self.path,
                                         path_params, alphas=this_alphas,
                                         l1_ratio=this_l1_ratio, X_order='F',
                                         dtype=X.dtype.type)
                for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
                for train, test in folds)
        mse_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer="threads"))(jobs)
        mse_paths = np.reshape(mse_paths, (n_l1_ratio, len(folds), -1))
        mean_mse = np.mean(mse_paths, axis=1)
        self.mse_path_ = np.squeeze(np.rollaxis(mse_paths, 2, 1))
        for l1_ratio, l1_alphas, mse_alphas in zip(l1_ratios, alphas,
                                                   mean_mse):
            i_best_alpha = np.argmin(mse_alphas)
            this_best_mse = mse_alphas[i_best_alpha]
            if this_best_mse < best_mse:
                best_alpha = l1_alphas[i_best_alpha]
                best_l1_ratio = l1_ratio
                best_mse = this_best_mse

        self.l1_ratio_ = best_l1_ratio
        self.alpha_ = best_alpha
        if self.alphas is None:
            self.alphas_ = np.asarray(alphas)
            if n_l1_ratio == 1:
                self.alphas_ = self.alphas_[0]
        # Remove duplicate alphas in case alphas is provided.
        else:
            self.alphas_ = np.asarray(alphas[0])

        # Refit the model with the parameters selected
        common_params = {name: value
                         for name, value in self.get_params().items()
                         if name in model.get_params()}
        model.set_params(**common_params)
        model.alpha = best_alpha
        model.l1_ratio = best_l1_ratio
        model.copy_X = copy_X
        precompute = getattr(self, "precompute", None)
        if isinstance(precompute, str) and precompute == "auto":
            model.precompute = False
        model.fit(X, y)
        if not hasattr(self, 'l1_ratio'):
            del self.l1_ratio_
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.dual_gap_ = model.dual_gap_
        self.n_iter_ = model.n_iter_
        return self
