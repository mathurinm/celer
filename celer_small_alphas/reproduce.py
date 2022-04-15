# data available at  https://www.dropbox.com/sh/32b3mr3xghi496g/AACNRS_NOsUXU-hrSLixNg0ja?dl=0


import time
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
from celer import GroupLasso

X = np.load("design_matrix.npy")
y = np.load("target.npy")
groups = np.load("groups.npy")
weights = np.load("weights.npy")
grps = [list(np.where(groups == i)[0]) for i in range(1, 33)]


alpha_ratio = 1e-2
n_alphas = 10


# Case 1: slower runtime for (very) small alphas
# alpha_max = 0.003471727067743962
# alpha_max = np.max(np.linalg.norm((X.T @ y).reshape(-1, 5), axis=1)) / len(y)
# grid = alpha_max * np.geomspace(alpha_ratio, 1, n_alphas)[::-1]
# times = []
# clf = GroupLasso(fit_intercept=False,
#                  groups=5, warm_start=True, verbose=1)
# for a in grid:
#     clf.alpha = a
#     t0 = time.time()
#     clf.fit(X, y)
#     t1 = time.time()
#     times.append(t1 - t0)
#     print(f"Finished tuning with {a:.2e}. Took {t1-t0:.2f} seconds!")

# fig, ax = plt.subplots()
# # hack semilogx bar width
# ax.bar(grid[1:] / alpha_max, times[1:],
#        width=-0.5 * np.diff(grid) / alpha_max, log=True)
# ax.set_ylabel("Time (s)")
# ax.set_xlabel("alpha/alpha_max")
# ax.set_xscale("log")
# plt.show(block=False)


# Case 2: slower runtime for (very) small alphas with weights
# alpha_max_w = 0.0001897719130007628
alpha_max_w = np.max(norm((X.T @ y).reshape(-1, 5) /
                          weights[:, None], axis=1)) / len(y)


alpha_ratio = 0.1
grid_w = np.geomspace(alpha_max_w*alpha_ratio, alpha_max_w, n_alphas)[::-1]
clf = GroupLasso(fit_intercept=False,
                 weights=weights, groups=grps, warm_start=True)

# for alpha in grid_w:
#     clf.alpha = alpha
#     t0 = time.time()
#     clf.fit(X, y)
#     t1 = time.time()
#     print(f"Finished tuning with {alpha:.2e}. Took {t1-t0:.2f} seconds!")


# Case 3.1 : (very) slow runtime when including a weight that is np.inf
weights[-1] = np.inf
clf = GroupLasso(fit_intercept=False,
                 weights=weights, groups=grps, warm_start=True, verbose=2)
for alpha in grid_w:
    clf.alpha = alpha
    t0 = time.time()
    clf.fit(X, y)
    t1 = time.time()
    print(f"Finished tuning with {alpha:.2e}. Took {t1-t0:.2f} seconds!")


1 / 0

# Case 3.2: remove np.inf from weights and extract elements of X and grps accordingly --> much faster than 3.1
weights = weights[:-1]
grps = grps[:-1]
X_new = X[:, :-5]
for a in grid_w:
    clf = GroupLasso(alpha=a, fit_intercept=False,
                     weights=weights, groups=grps, warm_start=True)
    t0 = time.time()
    clf.fit(X_new, y)
    t1 = time.time()
    print(f"Finished tuning with {a:.2e}. Took {t1-t0:.2f} seconds!")
