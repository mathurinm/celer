"""
===============================================================
ElasticNet vs. Lasso: Illustration of ElasticNet use case
===============================================================

The Lasso estimator is useful in features selection thanks
to its ability to produce sparse solutions. However,
the latter becomes ineffective when it comes to high dimensional
datasets as the support of the produced solution is limited
by the number of samples.

The ElasticNet estimator extend the feature selection property
of Lasso estimator by producing sparse solutions regardless
of the number of samples.

Below is an example that illustrates that.
"""
from celer import ElasticNet, Lasso

import numpy as np
from numpy.linalg import norm
from scipy.linalg import block_diag

import matplotlib.pyplot as plt


###############################################################################
# Create function to generate group-correlated data

def make_group_correlated_data(n_samples, n_features, n_groups, corr, snr,
                               random_state):
    if random_state is not None:
        np.random.seed(random_state)

    n_features_group = n_features // n_groups

    # build corr matrix
    blocs_corr_matrix = []
    for _ in range(n_groups):
        bloc_matrix = np.array([[corr]*i + [1] + [corr]*(n_features_group-i-1)
                                for i in range(n_features_group)], dtype=float)
        blocs_corr_matrix.append(bloc_matrix)

    corr_matrix = block_diag(*blocs_corr_matrix)

    # weight vector
    w_group = np.random.choice(2, n_groups)
    w_true = np.repeat(w_group, n_features_group)

    # build X, y
    mean_vec = np.zeros(n_features)
    X = np.random.multivariate_normal(mean_vec, corr_matrix, size=n_samples)
    y = X @ w_true

    if snr != np.inf:
        noise = np.random.randn(n_samples)
        y += noise / norm(noise) * norm(y) / snr

    return X, y, w_true


###############################################################################
#

random_state = 42
data_params = {'n_groups': 10, 'corr': 0.8, 'snr': 2, 'random_state': random_state}

n_samples, n_features = 1000, 100
X, y, w_true = make_group_correlated_data(n_samples, n_features, **data_params)
alpha_max = norm(X.T@y, ord=np.inf) / n_samples
alpha = alpha_max / 10.

# init/fit ElasticNet and Lasso
l1_ratio = 0.85
enet = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)
enet.fit(X, y)

lasso = Lasso(alpha=alpha)
lasso.fit(X, y)


###############################################################################
# Case 1: number of samples is much larger than number of features. In this
# setup, both estimators should produce similar results.

models = (('ElasticNet', enet), ('Lasso', lasso))
fig, axs = plt.subplots(len(models), 1)

for i, (model_name, model) in enumerate(models):
    # true coefs
    axs[i].stem(np.where(w_true)[0], w_true[w_true != 0], linefmt='g',
                label=r"true regression coefficients", markerfmt='go')
    # model coefs
    w = model.coef_
    axs[i].stem(np.where(w)[0], w[w != 0], linefmt='r', label=model_name,
                markerfmt='rx')

    axs[i].set_ylabel("%s coefs" % model_name)


axs[-1].set_xlabel("feature index")
fig.suptitle(
    r"Comparison between true (green) and estimated coefficients (case: n > p)")

print(
    f"Support size of Lasso: {len((lasso.coef_[lasso.coef_ != 0]))}\n"
    f"Support size of ElasticNet: {len(enet.coef_[enet.coef_ != 0])}"
)

plt.show()

###############################################################################
# Case 2: number of features is much larger than number of samples. In this
# setup, Lasso should fail to perform a good feature selection.

n_samples, n_features = 500, 1000
X, y, w_true = make_group_correlated_data(n_samples, n_features, **data_params)
alpha_max = norm(X.T@y, ord=np.inf) / n_samples
alpha = alpha_max / 10.

l1_ratio = 0.85
enet = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)
enet.fit(X, y)

lasso = Lasso(alpha=alpha)
lasso.fit(X, y)

# plot results
models = (('ElasticNet', enet), ('Lasso', lasso))
fig, axs = plt.subplots(len(models), 1)

for i, (model_name, model) in enumerate(models):
    # true coefs
    axs[i].stem(np.where(w_true)[0], w_true[w_true != 0], linefmt='g',
                label=r"true regression coefficients", markerfmt='go')
    # model coefs
    w = model.coef_
    axs[i].stem(np.where(w)[0], w[w != 0], linefmt='r', label=model_name,
                markerfmt='rx')

    axs[i].set_ylabel("%s coefs" % model_name)


axs[-1].set_xlabel("feature index")
fig.suptitle(
    r"Comparison between true (green) and estimated coefficients (case: n < p)")

print(
    f"Support size of Lasso: {len((lasso.coef_[lasso.coef_ != 0]))}\n"
    f"Support size of ElasticNet: {len(enet.coef_[enet.coef_ != 0])}"
)

plt.show()
