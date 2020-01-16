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

from celer import LassoCV

print(__doc__)

print("Loading data...")
dataset = fetch_openml("leukemia")
X = np.asfortranarray(dataset.data.astype(float))
y = 2 * ((dataset.target == "AML") - 0.5)
n_samples = len(y)

model = LassoCV(cv=3)
model.fit(X, y)

print("Estimated regularization parameter alpha: %s" % model.alpha_)

# Display results
m_log_alphas = -np.log10(model.alphas_)

plt.figure()
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.axis('tight')
plt.show()
