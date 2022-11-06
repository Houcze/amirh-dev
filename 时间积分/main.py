import numpy as np
import matplotlib.pyplot as plt

def f(x, t):
    return 1

def k1(func, t_n, y_n, h):
    return func(t_n, y_n)

def k2(func, t_n, y_n, h):
    return func(t_n + h / 2, y_n + k1(func, t_n, y_n, h) * h / 2)

def k3(func, t_n, y_n, h):
    return func(t_n + h / 2, y_n + k2(func, t_n, y_n, h) * h / 2)

def k4(func, t_n, y_n, h):
    return func(t_n + h, y_n + k3(func, t_n, y_n, h) * h)

def next(func, y_n, t_n, h):
    return y_n + (1 / 6) * (k1(func, t_n, y_n, h) + k2(func, t_n, y_n, h) + k3(func, t_n, y_n, h) + k4(func, t_n, y_n, h)) * h


t0 = 0
y0 = 1
y = y0
h = 0.1

index = []
result = []

for i in range(100):
    y = next(f, y, t0 + i * h, h)
    index.append(t0 + i * h)
    result.append(y)

plt.plot(index, result)
plt.savefig('result.jpg')

