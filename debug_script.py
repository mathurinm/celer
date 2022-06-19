# imports
from sklearn.linear_model import LassoCV as sk_LassoCV
from celer import LassoCV
import time
from celer.datasets import make_correlated_data
from sklearn.model_selection import train_test_split

# generate the toy dataset
X, y, _ = make_correlated_data(n_samples=500, n_features=5000)
# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# imports
start = time.time()
celer_lassoCV = LassoCV(n_alphas=100, cv=5)
celer_lassoCV.fit(X, y)
print(f"time elapsed for celer LassoCV: {time.time() - start}")

start = time.time()
sk_lassoCV = sk_LassoCV(n_alphas=100, cv=5)
sk_lassoCV.fit(X, y)
print(f"time elapsed for scikit-learn LassoCV: {time.time() - start}")
