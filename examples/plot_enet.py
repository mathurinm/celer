"""
=============================================================
Run ElasticNetCV for cross-validation on the Leukemia dataset
=============================================================

The example runs the ElasticNetCV scikit-learn like estimator
using the Celer algorithm.
"""

import matplotlib.pyplot as plt

from celer.datasets.simulated import make_correlated_data
from sklearn.model_selection import KFold

from celer import ElasticNetCV
from celer.plot_utils import configure_plt

print(__doc__)
configure_plt()


X, y, _ = make_correlated_data(50, 100, corr=.9, random_state=0)

n_jobs = 5
kf = KFold(shuffle=True, n_splits=3, random_state=0)
model = ElasticNetCV(cv=kf, n_jobs=n_jobs, l1_ratio=0.8)
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

model = ElasticNetCV(cv=kf, n_jobs=n_jobs, l1_ratio=l1_ratios)
model.fit(X, y)

print("Estimated regularization parameter alpha: %s" % model.alpha_)
print("Best mix penalty parameter (l1_ratio): %s" % model.l1_ratio_)

###############################################################################
# Plot of the CV results

for i in range(len(l1_ratios)):
    alphas = model.alphas_[i]
    mse_path = model.mse_path_[i]

    mean_mse_path = mse_path.mean(axis=-1)
    std_mse_path = mse_path.std(axis=-1)

    plt.semilogx(alphas, mean_mse_path, linewidth=1, label='L1_ratio=%s' % l1_ratios[i])
    plt.fill_between(
        x=alphas,
        y1=mean_mse_path + std_mse_path,
        y2=mean_mse_path - std_mse_path,
        alpha=0.2,
    )

plt.axvline(model.alpha_, linestyle='--', color='k')

plt.ylabel('Mean square prediction error')
plt.xlabel(r'$\alpha$')
plt.legend()
plt.title(r"ElasticNetCV with many l1_ratios and $\alpha$")

plt.show()
