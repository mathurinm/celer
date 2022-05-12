import numpy as np
from scipy.linalg import block_diag

from celer import ElasticNetCV, LassoCV
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


def make_group_correlated_data(n_samples, n_features, n_groups,
                               corr=0.8, snr=.3, return_w_group=False,
                               random_state=None):
    """Generate X, y dataset such that X[:, j] are iid and follow N(0, corr_matrix)
    where corr_matrix is bloc diagonal matrix with bloc
    B[i, j] = corr for i != j and 1 otherwise."""
    if random_state is not None:
        np.random.seed(random_state)

    n_features_group = n_features // n_groups

    # build corr matrix
    blocs_corr_matrix = []
    for g in range(n_groups):
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
    noise = np.random.randn(n_samples)
    y += noise / np.linalg.norm(noise) * np.linalg.norm(y) / snr

    if return_w_group:
        return X, y, w_true, w_group

    return X, y, w_true


def group_effect_score(w_true, w_estimated):
    """Ratio of in/out features."""
    n_features = len(w_true)

    # count coefs in one groups
    one_groups_coefs = w_estimated[w_true.astype(bool)]
    n_non_zero_coefs = len(one_groups_coefs[one_groups_coefs != 0])

    # count coefs in zero groups
    zero_groups_coefs = w_estimated[~ w_true.astype(bool)]
    n_zero_coefs = len(zero_groups_coefs[zero_groups_coefs == 0])

    return (n_non_zero_coefs + n_zero_coefs) / n_features


if __name__ == '__main__':
    random_state = 42
    n_samples, n_features, n_groups, corr, snr = 100, 1000, 10, 0.8, 3

    X, y, w_true = make_group_correlated_data(
        n_samples, n_features, n_groups, corr, snr, random_state=random_state)

    # fit model params
    eps, n_alphas = 1e-3, 100
    alpha_max = np.linalg.norm(X.T@y, ord=np.inf) / n_samples
    alphas = alpha_max * np.geomspace(1, eps, n_alphas)
    kf = KFold(n_splits=5, random_state=random_state, shuffle=True)

    # fit ElasticNet
    print("fitting ElasticNet...")
    enet = ElasticNetCV(l1_ratio=[0.75, 0.8, 0.85, 0.9, 0.95],
                        cv=kf, n_jobs=5, alphas=alphas)
    enet.fit(X, y)

    # fit lasso
    print("fitting Lasso...")
    lasso = LassoCV(cv=kf, n_jobs=5, alphas=alphas)
    lasso.fit(X, y)

    # plot
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    m, s, _ = plt.stem(np.where(w_true)[0], w_true[w_true != 0],
                       label=r"true regression coefficients",
                       use_line_collection=True)
    labels = ["ENET coefs"]
    colors = [u'#ff7f0e']

    for w, label, color in zip([enet.coef_], labels, colors):
        m, s, _ = plt.stem(np.where(w)[0], w[w != 0], label=label,
                           markerfmt='x', use_line_collection=True)
        plt.setp([m, s], color=color)

    score_lasso = group_effect_score(w_true, lasso.coef_)
    score_enet = group_effect_score(w_true, enet.coef_)
    plt.title(f"Lasso: {score_lasso:.2f} vs. ElasticNet: {score_enet:.2f}")
    plt.show()
