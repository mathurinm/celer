import time
import numpy as np
from numpy.linalg import norm

from celer import Lasso
from sklearn.linear_model import Lasso as sk_Lasso
from sklearn.datasets import fetch_openml

from benchopt.datasets.simulated import make_correlated_data
import matplotlib.pyplot as plt


def process_leukemia():
    dataset = fetch_openml("leukemia")
    X = np.asfortranarray(dataset.data.astype(float))
    y = 2 * ((dataset.target == "AML") - 0.5)
    y -= np.mean(y)
    y /= np.std(y)
    return X, y


class Comparison:
    n_reps = 100

    def __init__(self, models) -> None:
        self.models = models
        self.dict_exc_times_ = {name: None for name in models}

    def time(self, X, y):
        for name, model in self.models.items():
            exc_time = Comparison.compute_time(
                lambda: model.fit(X, y),
            )
            self.dict_exc_times_[name] = exc_time
        return self

    @staticmethod
    def compute_time(func):
        sum_duration = 0.

        for _ in range(Comparison.n_reps):
            start = time.time()
            func()
            sum_duration += time.time() - start

        return sum_duration / Comparison.n_reps


RATIO = 1e-2
params = {'tol': 1e-14, 'fit_intercept': True, }
X, y = process_leukemia()
alpha = RATIO * norm(X.T @ y, ord=np.inf)

compare_leukemia = Comparison(models={
    'celer': Lasso(alpha=alpha, **params),
    'sklearn': sk_Lasso(alpha=alpha, **params),
}).time(X, y)

X, y, _ = make_correlated_data(50, 100, random_state=0)
alpha = RATIO * norm(X.T @ y, ord=np.inf)

compare_correlated = Comparison(models={
    'celer': Lasso(alpha=alpha, **params),
    'sklearn': sk_Lasso(alpha=alpha, **params)
}).time(X, y)


comparisons = (compare_leukemia, compare_correlated)
group_labels = ['leukemia', 'correlated']

x = np.arange(len(group_labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, [comp.dict_exc_times_['celer']
                for comp in comparisons], width, label='celer')
rects2 = ax.bar(x + width/2, [comp.dict_exc_times_['sklearn']
                for comp in comparisons], width, label='sklearn')

ax.set_ylabel('Times in second')
ax.set_title('ElasticNet branch')
ax.set_xticks(x, group_labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
