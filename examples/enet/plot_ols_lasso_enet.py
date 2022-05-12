import numpy as np
import matplotlib.pyplot as plt

from celer import ElasticNetCV, LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from examples.enet.tools import make_group_correlated_data

random_state = 42
n_samples, n_features, n_groups, corr, snr = 500, 1000, 10, 0.8, 3

X, y, w_true = make_group_correlated_data(
    n_samples, n_features, n_groups, corr, snr, random_state)

# fit model params
eps, n_alphas = 1e-3, 100
alpha_max = np.linalg.norm(X.T@y, ord=np.inf) / n_samples
alphas = alpha_max * np.geomspace(1, eps, n_alphas)
kf = KFold(n_splits=5, random_state=random_state, shuffle=True)

# fit ElasticNet
print("fitting ElasticNet...")
enet = ElasticNetCV(l1_ratio=[0.8],
                    cv=kf, n_jobs=5, alphas=alphas)
enet.fit(X, y)

# fit lasso
print("fitting Lasso...")
lasso = LassoCV(cv=kf, n_jobs=5, alphas=alphas)
lasso.fit(X, y)

# OLS
print("fitting OLS...")
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
fig.suptitle(
    r"Comparison of regression coefficients case p >> n (green are the true coef)")
plt.show()
