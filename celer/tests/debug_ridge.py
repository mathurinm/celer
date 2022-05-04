from celer import ElasticNet
from sklearn.linear_model import ElasticNet as sk_ElasticNet
from celer.utils.testing import build_dataset


n_samples, n_features = 30, 50
X, y = build_dataset(n_samples, n_features)


reg = ElasticNet(l1_ratio=1e-9, tol=1e-8, fit_intercept=True, verbose=2)
reg.fit(X, y)


reg = sk_ElasticNet(l1_ratio=1e-9, tol=1e-8, fit_intercept=True, verbose=2)
reg.fit(X, y)
