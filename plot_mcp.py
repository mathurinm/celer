from scipy.optimize import minimize
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True


def mcp_pen(x, lmbda, gamma):
    x = np.abs(x)
    if x < gamma * lmbda:
        return lmbda * x - x ** 2 / (2. * gamma)
    else:
        return lmbda ** 2 * gamma / 2


def prox_obj(x, y, lmbda, gamma, alpha=1):
    return (x - y) ** 2 / 2. + alpha * mcp_pen(y, lmbda, gamma)


lmbda = 1.5
gamma = 3
x = np.linspace(0, 7, 200)
y = np.array([mcp_pen(xi, lmbda, gamma) for xi in x])


plt.close('all')
plt.figure()

plt.axvline(lmbda, label=r'$\lambda$',
            color='k', linestyle='dotted')
plt.axvline(lmbda * gamma, label=r'$\lambda \gamma$',
            color='k', linestyle='--')

plt.title(r'$\lambda, \gamma = (%s, %s)$' % (lmbda, gamma))
plt.plot(
    x, y, label=r'$FT(x, \lambda, \gamma)$')

for alpha in [1, 2, 3, 3.1, 4]:
    prox_values = []
    for xi in x:
        res = minimize(lambda y: prox_obj(
            xi, y, lmbda, gamma, alpha=alpha), x0=0)
        prox_values.append(res.x)

    plt.plot(x, prox_values,
             label=r'$\mathrm{prox}_{\alpha FT(\cdot, \lambda, \gamma)}(x) \quad \alpha = %s$' % (
                 alpha))


plt.xlabel('$x$')
plt.legend()
plt.show(block=False)
