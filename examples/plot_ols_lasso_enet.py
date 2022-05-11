import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

from celer import ElasticNetCV, LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


n_samples, n_features, n_groups, corr, snr = 10000, 200, 10, 0.8, 2
n_features_group = n_features // n_groups

# build corr matrix
bloc_corr_matrices = []

for _ in range(n_groups):
    bloc_matrix = np.array([[corr]*i + [1] + [corr]*(n_features_group-i-1)
                            for i in range(n_features_group)], dtype=float)

    bloc_corr_matrices.append(bloc_matrix)

corr_matrix = block_diag(*bloc_corr_matrices)

# generate data
random_state = 42
np.random.seed(random_state)

weight_group = np.random.choice(2, n_groups)
w_true = np.repeat(weight_group, n_features_group)

mean_vec = np.zeros(n_features)
X = np.random.multivariate_normal(mean_vec, corr_matrix, size=n_samples)
y = X @ w_true

noise = np.random.randn(n_samples)
y += noise / np.linalg.norm(noise) * np.linalg.norm(y) / snr

# fit model params
eps, n_alphas = 1e-3, 100
alpha_max = np.linalg.norm(X.T@y, ord=np.inf) / n_samples
alphas = alpha_max * np.geomspace(1, eps, n_alphas)
kf = KFold(n_splits=5, random_state=random_state, shuffle=True)

# fit ElasticNet
enet = ElasticNetCV(l1_ratio=[0.8],
                    cv=kf, n_jobs=5, alphas=alphas)
enet.fit(X, y)

# fit lasso
lasso = LassoCV(cv=kf, n_jobs=5, alphas=alphas)
lasso.fit(X, y)

# OLS
ols = LinearRegression()
ols.fit(X, y)

# plot coefs
fig, axs = plt.subplots(3, 1)

for i, (model_name, model) in enumerate(
        (('OLS', ols), ('Lasso', lasso), ('ElasticNet', enet))):
    # true coefs
    axs[i].stem(np.where(w_true)[0], w_true[w_true != 0], 'g',
                label=r"true regression coefficients", markerfmt='go')
    # model coefs
    w = model.coef_
    axs[i].stem(np.where(w)[0], w[w != 0], 'r', label=model_name,
                markerfmt='rx')

    # axs[i].set_title("%s coefficients" % model_name)
    axs[i].set_ylabel("%s coefs" % model_name)


axs[-1].set_xlabel("feature index")
fig.suptitle("Comparison of regression coefficients case p >> n (green are the true coef)")
plt.show()
