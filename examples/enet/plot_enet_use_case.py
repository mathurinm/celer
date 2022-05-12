"""
===============================================================
ElasticNet vs. Lasso: Illustration of ElasticNet use case
===============================================================

Lasso estimator is useful in feature selection thanks to its
ability to produce sparse solutions. However, the latter
becomes ineffective when it comes to high dimensional data
as the support of the produced solution is limited by the
number of samples.

ElasticNet estimator extend the Lasso estimator by producing
sparse solutions regardless of the number of samples.
The following example illustrates such a property.
"""

from celer import ElasticNetCV, LassoCV
from sklearn.model_selection import KFold
from celer.datasets.simulated import make_group_correlated_data

import numpy as np
import matplotlib.pyplot as plt


random_state = 42
data_params = {'n_groups': 10, 'corr': 0.8, 'snr': 2, 'random_state': random_state}

n_samples, n_features = 1000, 100
X, y, w_true = make_group_correlated_data(n_samples, n_features, **data_params)
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)


# init and crossvalidate ElasticNet and Lasso
n_jobs = 5
l1_ratios = [0.8, 0.85, 0.9, 0.95]
enet = ElasticNetCV(l1_ratio=l1_ratios, cv=kf, n_jobs=n_jobs)
enet.fit(X, y)

lasso = LassoCV(cv=kf, n_jobs=n_jobs)
lasso.fit(X, y)


###############################################################################
# Case 1: number of samples is much larger than number of features. In this
# setup, both estimators should produce similar results.

models = (('ElasticNet', enet), ('Lasso', lasso))
fig, axs = plt.subplots(len(models), 1)

for i, (model_name, model) in enumerate(models):
    # true coefs
    axs[i].stem(np.where(w_true)[0], w_true[w_true != 0], 'g',
                label=r"true regression coefficients", markerfmt='go')
    # model coefs
    w = model.coef_
    axs[i].stem(np.where(w)[0], w[w != 0], 'r', label=model_name,
                markerfmt='rx')

    axs[i].set_ylabel("%s coefs" % model_name)


axs[-1].set_xlabel("feature index")
fig.suptitle(
    r"Comparison between true (green) and estimated coefficients (case: n > p)")

print(
    f"Support size of Lasso: {len((lasso.coef_[lasso.coef_ != 0]))}\n"
    f"Support size of ElasticNet: {len(enet.coef_[enet.coef_ != 0])}\n"
)

plt.show()

###############################################################################
# Case 2: number of features is much larger than number of samples. In this
# setup, Lasso should fail to perform a good feature selection.

n_samples, n_features = 200, 1000
X, y, w_true = make_group_correlated_data(n_samples, n_features, **data_params)

enet = ElasticNetCV(l1_ratio=l1_ratios, cv=kf, n_jobs=n_jobs)
enet.fit(X, y)

lasso = LassoCV(cv=kf, n_jobs=n_jobs)
lasso.fit(X, y)

# plot results
models = (('ElasticNet', enet), ('Lasso', lasso))
fig, axs = plt.subplots(len(models), 1)

for i, (model_name, model) in enumerate(models):
    # true coefs
    axs[i].stem(np.where(w_true)[0], w_true[w_true != 0], 'g',
                label=r"true regression coefficients", markerfmt='go')
    # model coefs
    w = model.coef_
    axs[i].stem(np.where(w)[0], w[w != 0], 'r', label=model_name,
                markerfmt='rx')

    axs[i].set_ylabel("%s coefs" % model_name)


axs[-1].set_xlabel("feature index")
fig.suptitle(
    r"Comparison between true (green) and estimated coefficients (case: n < p)")

print(
    f"Support size of Lasso: {len((lasso.coef_[lasso.coef_ != 0]))}\n"
    f"Support size of ElasticNet: {len(enet.coef_[enet.coef_ != 0])}\n"
)

plt.show()
