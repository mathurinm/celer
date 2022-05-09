"""
=============================================================
Run ElasticNetCV for cross-validation on the Leukemia dataset
=============================================================

The example runs the ElasticNetCV scikit-learn like estimator
using the Celer algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

from benchopt.datasets.simulated import make_correlated_data
from sklearn.model_selection import KFold

from celer import ElasticNetCV
from celer.plot_utils import configure_plt

print(__doc__)
configure_plt()


X, y, _ = make_correlated_data(50, 100, rho=.9, random_state=0)

kf = KFold(shuffle=True, n_splits=3, random_state=0)
model = ElasticNetCV(cv=kf, n_jobs=3, l1_ratio=0.8)
model.fit(X, y)

print("Estimated regularization parameter alpha: %s" % model.alpha_)

###############################################################################
# Display results

mean_mse_path = model.mse_path_.mean(axis=-1)
std_mse_path = model.mse_path_.std(axis=-1)

plt.figure(figsize=(5, 3), constrained_layout=True)

plt.semilogx(model.alphas_, mean_mse_path, 'k',
             label='Average across the folds', linewidth=2)
plt.axvline(model.alpha_, linestyle='--', color='k',
            label='alpha: CV estimate')

plt.fill_between(
    x=model.alphas_,
    y1=mean_mse_path + std_mse_path,
    y2=mean_mse_path - std_mse_path,
    alpha=0.5,
    label="Standard deviation across the folds",
)

plt.legend()
plt.xlabel(r'$\alpha$')
plt.xscale('log')
plt.ylabel('Mean square prediction error')
plt.show()

###############################################################################
# It is also possible to crossvalidate the model using a list of l1_ratios

l1_ratios = [1., 0.9, 0.8, 0.7]

model = ElasticNetCV(cv=kf, n_jobs=5, l1_ratio=l1_ratios)
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

    mean_mse_path = mse_path.mean(axis=-1)
    std_mse_path = mse_path.std(axis=-1)
    arg_min = np.argmin(mean_mse_path, axis=None)
    min_mse = mean_mse_path[arg_min]

    axs[i].semilogx(alphas, mean_mse_path, 'k', linewidth=2)
    axs[i].axvline(alphas[arg_min], linestyle='--', color='k')
    axs[i].fill_between(
        x=alphas,
        y1=mean_mse_path + std_mse_path,
        y2=mean_mse_path - std_mse_path,
        alpha=0.5,
    )

    axs[i].set_xlabel(r'$\alpha$')
    axs[i].set_title(f'l1_ratio = {l1_ratios[i]} (best mse={min_mse:.2f})',
                     fontsize=10)


axs[0].set_ylabel('Mean square prediction error')
fig.suptitle(r"ElasticNetCV with many l1_ratios and $\alpha$")
fig.tight_layout()
plt.show()
