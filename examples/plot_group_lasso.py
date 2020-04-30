"""
=============================================
Run GroupLasso for structured sparse recovery
=============================================

The example runs the GroupLasso scikit-learn like estimators.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state

from celer import GroupLasso
from celer.plot_utils import configure_plt

print(__doc__)
configure_plt()

# Generating X and y data

n_samples, n_features = 30, 50
rng = check_random_state(0)
X = rng.randn(n_samples, n_features)


# Create true regression coefficients with 3 groups of 5 non-zero values

w_true = np.zeros(n_features)
w_true[:5] = 1
w_true[20:25] = -2
w_true[40:45] = 1
y = X @ w_true + rng.randn(n_samples)


# Fit an adapted GroupLasso model

groups = 5  # groups are contiguous and of size 5
clf = GroupLasso(groups=groups, alpha=1)
clf.fit(X, y)

# Display results

fig = plt.figure(figsize=(13, 4))
m, s, _ = plt.stem(w_true, label=r"true regression coefficients")
m, s, _ = plt.stem(clf.coef_, label=r"estimated regression coefficients",
                   markerfmt='x')
plt.setp([m, s], color='#ff7f0e')
plt.xlabel("feature index")
plt.legend()
plt.show(block=False)
