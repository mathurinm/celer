from celer import Lasso

X = [[1, 0], [0, 1]]
y = [1, 1]

Lasso().fit(X, y)
