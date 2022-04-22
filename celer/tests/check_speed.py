import time
import numpy as np
import matplotlib.pyplot as plt


from celer import Lasso, LogisticRegression, MultiTaskLasso
from sklearn.linear_model import (Lasso as sk_Lasso,
                                  LogisticRegression as sk_LogisticRegression,
                                  MultiTaskLasso as sk_MultiTaskLasso,
                                  )

from celer.utils.testing import build_dataset


def fit_and_time(model, X, y, is_logreg=False):
    n_fits = 10
    alpha_max = np.linalg.norm(X.T.dot(y), ord=np.inf)

    arr_times = np.zeros(n_fits)
    for i in range(n_fits):
        start = time.time()
        if is_logreg:
            m = model(C=100. / alpha_max)
        else:
            m = model(alpha=alpha_max / 100.)
        m.fit(X, y)
        end = time.time()

        arr_times[i] = end - start

    return arr_times.mean()


arr_n_features = 100 * np.arange(2, 10 + 1)
n_sample = 100

dict_times = {
    'Lasso': {'celer': [], 'sklearn': []},
    'LogReg': {'celer': [], 'sklearn': []},
    'Mlt': {'celer': [], 'sklearn': []},
}

for n_features in arr_n_features:
    X, y = build_dataset(n_sample, n_features)
    dict_times['Lasso']['celer'].append(fit_and_time(Lasso, X, y))
    dict_times['Lasso']['sklearn'].append(fit_and_time(sk_Lasso, X, y))

    X, y = build_dataset(n_sample, n_features)
    y = np.sign(y)
    dict_times['LogReg']['celer'].append(
        fit_and_time(LogisticRegression, X, y, is_logreg=True))
    dict_times['LogReg']['sklearn'].append(
        fit_and_time(sk_LogisticRegression, X, y, is_logreg=True))

    X, y = build_dataset(n_sample, n_features, n_targets=10)
    dict_times['Mlt']['celer'].append(fit_and_time(MultiTaskLasso, X, y))
    dict_times['Mlt']['sklearn'].append(fit_and_time(sk_MultiTaskLasso, X, y))


fig, ax = plt.subplots(1, 3)

for i, (model_name, dict_packs_times) in enumerate(dict_times.items()):
    for pack_name, arr_times in dict_packs_times.items():
        ax[i].plot(
            arr_n_features,
            arr_times,
            label=pack_name,
        )

    ax[i].set_title(model_name)
    ax[i].set_xlabel('n_features')
    ax[i].set_ylabel('time (s)')
    ax[i].legend()


fig.suptitle("Updated version")

plt.tight_layout()
plt.show()
