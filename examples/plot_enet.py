"""
=============================================================
Run ElasticNetCV for cross-validation on the Leukemia dataset
=============================================================

The example runs the ElasticNetCV scikit-learn like estimator
using the Celer algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

from celer import ElasticNetCV
from celer.plot_utils import configure_plt

print(__doc__)
configure_plt()

print("Loading data...")
dataset = fetch_openml("leukemia")
X = np.asfortranarray(dataset.data.astype(float))
y = 2 * ((dataset.target == "AML") - 0.5)
y -= np.mean(y)
y /= np.std(y)

kf = KFold(shuffle=True, n_splits=3, random_state=0)
model = ElasticNetCV(cv=kf, n_jobs=3)
model.fit(X, y)

print("Estimated regularization parameter alpha: %s" % model.alpha_)

###############################################################################
# Display results

plt.figure(figsize=(5, 3), constrained_layout=True)
plt.semilogx(model.alphas_, model.mse_path_, ':')
plt.semilogx(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
plt.axvline(model.alpha_, linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel(r'$\alpha$')
plt.ylabel('Mean square prediction error')
plt.show(block=False)

###############################################################################
# It is also possible to crossvalidate the model using a list of l1_ratios

l1_ratios = np.array([0.8, 0.9])

model = ElasticNetCV(cv=kf, n_jobs=3, l1_ratio=l1_ratios)
model.fit(X, y)

print("Estimated regularization parameter alpha: %s" % model.alpha_)
print("Best mix penalty parameter (l1_ratio): %s" % model.l1_ratio_)

###############################################################################
# Plot of the CV results

n_figures = len(l1_ratios)
fig, axs = plt.subplots(1, n_figures)

for i in range(n_figures):
    alphas = model.alphas_[i]
    mse_path = model.mse_path_[i]

    axs[i].semilogx(alphas, mse_path, ':')
    axs[i].semilogx(alphas, mse_path.mean(axis=-1), 'k',
                    label='Average across the folds', linewidth=2)
    axs[i].axvline(model.alpha_, linestyle='--', color='k',
                   label='alpha: CV estimate')

    axs[i].set_ylabel('Mean square prediction error')
    axs[i].set_xlabel(r'$\alpha$')
    axs[i].legend()
    axs[i].set_title('l1_ratio = %s' % l1_ratios[i])

fig.suptitle(r"ElasticNetCV with both l1_ratio and $\alpha$")
fig.tight_layout()
