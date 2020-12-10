# %%
import random
import numpy as np
import matplotlib.pyplot as plt
from math import *

x = []
y = []
for i in range(100):
    x0 = random.uniform(-4*pi, 4*pi)
    y0 = sin(0.4*x0) + random.gauss(0, 0.1)
    x.append(x0)
    y.append(y0)
plt.plot(x, y, '.')
plt.show()


def f(x, k):
    return sin(k*x)


bestLoss = 1e9
best_k = 0
dt = 0.001
lossHistory = []
kHistory = []
k = 0.2
for i in range(5000):
    kGrad = [2 * (y[i] - f(x[i], k)) * x[i] * cos(k*x[i])
             for i in range(len(y))]
    k += sum(kGrad)/len(kGrad)*dt
    # print(a,b)

    reses = [(y[i] - f(x[i], k))**2 for i in range(len(y))]
    loss = sqrt(sum(reses) / len(reses))
    lossHistory.append(loss)
    kHistory.append(k)

    if loss < bestLoss:
        best_k = k
        bestLoss = loss
print(f"Best Loss : {bestLoss} \nBest k : {best_k}")


x0, x1 = min(x), max(x)
plt.plot(x, y, '.')
x = sorted(x)
plt.plot(x, [f(i, best_k) for i in x], '-')
plt.show()
# %%
