# %%
import random
import numpy as np
import matplotlib.pyplot as plt
from math import *

x = []
y = []
for i in range(100):
    x0 = random.uniform(-4*pi, 4*pi)
    y0 = 3 * sin(0.4*x0) + random.gauss(0, 0.1)
    x.append(x0)
    y.append(y0)
plt.plot(x, y, '.')
plt.show()


def f(x, A, k):
    return A * sin(k * x)


best_k = 0
bestA = 0
bestLoss = 1e9

dt = 0.001
lossHistory = []
kHistory = []
AHistory = []

k = 0.2
A = 5

for i in range(10000):
    kGrad = [-2 * (y[i] - f(x[i], A, k)) * A * x[i] * cos(k*x[i])
             for i in range(len(y))]
    AGrad = [-2 * (y[i] - f(x[i], A, k)) * sin(k * x[i])
             for i in range(len(y))]

    k -= sum(kGrad) / len(kGrad) * dt
    A -= sum(AGrad) / len(AGrad) * dt

    reses = [(y[i] - f(x[i], A, k)) ** 2 for i in range(len(y))]
    loss = sqrt(sum(reses) / len(reses))

    kHistory.append(k)
    AHistory.append(A)
    lossHistory.append(loss)

    if loss < bestLoss:
        best_k = k
        bestA = A
        bestLoss = loss

print(best_k, bestA, bestLoss)


x0, x1 = min(x), max(x)
plt.plot(x, y, '.')
x = sorted(x)
plt.plot(x, [f(i, bestA, best_k) for i in x], '-')
plt.show()
# %%
