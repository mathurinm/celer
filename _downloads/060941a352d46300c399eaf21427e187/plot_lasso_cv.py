"""
=======================================================
Run LassoCV for cross-validation on Leukemia dataset
=======================================================

The example runs the LassoCV scikit-learn like estimator
using the Celer algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

from celer import LassoCV
from celer.plot_utils import configure_plt

print(__doc__)
configure_plt()

print("Loading data...")
dataset = fetch_openml("leukemia")
X = np.asfortranarray(dataset.data.astype(float))
y = 2 * ((dataset.target == "AML") - 0.5)

kf = KFold(shuffle=True, n_splits=3, random_state=0)
model = LassoCV(cv=kf, n_jobs=3)
model.fit(X, y)

print("Estimated regularization parameter alpha: %s" % model.alpha_)

###############################################################################
# Display results

plt.figure(figsize=(7, 3.5), constrained_layout=True)
plt.semilogx(model.alphas_, model.mse_path_, ':')
plt.semilogx(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
plt.axvline(model.alpha_, linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel(r'$\alpha$')
plt.ylabel('Mean square error')
plt.show()
