"""
===================================
Compare LassoCV and AdaptiveLassoCV
===================================

Compare prediction and estimation performance of both models.
"""
import matplotlib.pyplot as plt

from celer import AdaptiveLassoCV, LassoCV
from celer.datasets import make_correlated_data

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from libsvmdata import fetch_libsvm

# X, y = fetch_libsvm("rcv1_train")

X, y, w_true = make_correlated_data(n_samples=600, n_features=2000, corr=0.6,
                                    density=0.1, snr=3)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True)

lasso = LassoCV(verbose=1, cv=4, eps=1e-3, n_jobs=-1).fit(X_train, y_train)
adalasso = AdaptiveLassoCV(verbose=1, cv=4, eps=1e-3,
                           n_jobs=-1).fit(X_train, y_train)


fig, axarr = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True,
                          sharey=True)
for i, model in enumerate([lasso, adalasso]):
    axarr[i].semilogx(model.alphas_, model.mse_path_, ':')
    axarr[i].semilogx(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
                      label='Average across the folds', linewidth=2)
    axarr[i].axvline(model.alpha_, linestyle='--', color='k',
                     label='alpha: CV estimate')

    axarr[i].legend()
    axarr[i].set_title(
        f"{model.__class__.__name__}, "
        f"$||\\hat w||_0 = {(model.coef_!=0).sum()}$")

    axarr[i].set_xlabel(r'$\alpha$')
    axarr[i].set_ylabel('Mean square prediction error')
plt.show(block=False)


print("Train:")
print("Lasso:", mean_squared_error(y_train, lasso.predict(X_train)))
print("Adalasso:", mean_squared_error(y_train, adalasso.predict(X_train)))
print("Test:")
print("Lasso:", mean_squared_error(y_test, lasso.predict(X_test)))
print("Adalasso:", mean_squared_error(y_test, adalasso.predict(X_test)))
