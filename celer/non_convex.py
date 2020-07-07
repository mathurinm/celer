import numpy as np

from numba import njit


@njit
def prox_mcp(x, alpha, gamma):
    absx = np.abs(x)
    if absx < alpha:
        return 0.
    elif absx < gamma * alpha:
        return np.sign(x) * gamma * (absx - alpha) / (gamma - 1.)
    else:
        return x


@njit
def primal_mcp(R, alpha, gamma, w):
    p_obj = (R ** 2).sum() / (2 * len(R))
    for j in range(len(w)):
        if abs(w[j]) < gamma * alpha:
            continue
        else:
            p_obj += gamma * alpha ** 2 / 2.
    return p_obj


@njit
def mcp(X, y, alpha, gamma, w_init, max_iter, verbose, tol):
    n_samples, n_features = X.shape
    norms = (X ** 2).sum(axis=0)
    w = np.copy(w_init)
    R = y - X @ w
    E = []
    E_freq = 10
    for t in range(max_iter):
        for j in range(n_features):
            w_old = w[j]
            w[j] = prox_mcp(w[j] + X[:, j] @ R / norms[j],
                            alpha * n_samples / norms[j], gamma)
            if w_old != w[j]:
                R += (w_old - w[j]) * X[:, j]
        # TODO KKT/primal stopping crit
        if t % E_freq == 0:
            p_obj = primal_mcp(R, alpha, gamma, w)
            E.append(p_obj)
            if verbose:
                print("Iter", t, "p_obj", p_obj)
    return w, E


if __name__ == "__main__":
    from numpy.linalg import norm
    np.random.seed(12)
    X = np.random.randn(100, 200)
    X = np.asfortranarray(X)
    w_star = np.zeros(X.shape[1])
    w_star[:30] = np.random.randn(30)
    y = X @ w_star
    noise = np.random.randn(len(X))
    y += 0.3 * noise * norm(y) / norm(noise)

    alpha_max_lasso = np.max(np.abs(X.T @ y)) / len(X)
    gamma = 1000
    alpha = alpha_max_lasso / 10

    _, E = mcp(X, y, alpha, gamma, np.zeros(X.shape[1]), 100, True, 1e-4)
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()

    from celer import MCP
    clf1 = MCP(alpha=alpha, gamma=gamma, verbose=1,
               max_iter=100, fit_intercept=False)
    clf1.fit(X, y)

    from celer import MCPCV
    clf2 = MCPCV(fit_intercept=False, gamma=3, n_jobs=-1).fit(X, y)

    from celer import LassoCV
    clf = LassoCV(n_jobs=-1, fit_intercept=False).fit(X, y)
    w_bis = clf.coef_

    for model in (clf, clf2):
        plt.figure()
        plt.semilogx(model.alphas_, model.mse_path_, ':')
        plt.semilogx(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
                     label='Average across the folds', linewidth=2)
        plt.axvline(model.alpha_, linestyle='--', color='k',
                    label='alpha: CV estimate')

        plt.legend()

        plt.xlabel(r'$\alpha$')
        plt.ylabel('Mean square error')
        plt.axis('tight')
        plt.title("%s, sparsity: %d" % (model.__class__.__name__,
                                        (model.coef_ != 0).sum()))
        plt.show(block=False)
