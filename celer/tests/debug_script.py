from numpy.linalg import norm
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

from celer import ElasticNetCV
from sklearn.linear_model import ElasticNetCV as sk_ElasticNetCV


dataset = fetch_openml("leukemia")
X = np.asfortranarray(dataset.data.astype(float))
y = 2 * ((dataset.target == "AML") - 0.5)
y -= np.mean(y)
y /= np.std(y)
kf = KFold(shuffle=True, n_splits=3, random_state=0)

params = {'cv': kf, 'l1_ratio': [0.8, 0.9], 'fit_intercept': True, 'n_jobs': 3}

# results for celer
model_celer = ElasticNetCV(**params)
model_celer .fit(X, y)

print("best l1_ratio: %f" % model_celer.l1_ratio_)
print("best mse: %f" % model_celer.mse_path_.min())
print(norm(model_celer.mse_path_[0] - model_celer.mse_path_[1], ord=np.inf))

print("***************************")

# results for sklearn
model_sk = sk_ElasticNetCV(**params)
model_sk.fit(X, y)

print("best l1_ratio: %f" % model_sk.l1_ratio_)
print("best mse: %f" % model_sk.mse_path_.min())
print(norm(model_sk.mse_path_[0] - model_sk.mse_path_[1], ord=np.inf))
