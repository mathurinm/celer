from sklearn.exceptions import ConvergenceWarning
import numpy as np
from numpy.linalg import norm
from sklearn.utils.estimator_checks import check_estimator

from celer.dropin_sklearn import LogisticRegression
from celer.utils.testing import build_dataset

import warnings
warnings.filterwarnings("error")


def isolate_logreg_warning(sparse_X):
    np.random.seed(1409)
    X, y = build_dataset(
        n_samples=30, n_features=60, sparse_X=sparse_X)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / 2
    C = 20. / alpha_max

    tol = 1e-4
    clf1 = LogisticRegression(C=C, tol=tol, verbose=0)

    generator = check_estimator(clf1, generate_only=True)

    dict_func_with_warn = {}
    for i, (es, check_func) in enumerate(generator):
        try:
            check_func(es)
        except ConvergenceWarning:
            dict_func_with_warn[i] = check_func
        except:
            pass

    print(10 * "*")
    print(
        f"{len(dict_func_with_warn)} out of {i+1} "
        "checks raise convergence warning"
    )
    print(10 * "*")

    print(dict_func_with_warn)


if __name__ == "__main__":
    isolate_logreg_warning(True)
