import numpy as np

N = 60000
a = np.ones((N, N))
b = np.zeros((N, N))

i = 1
j = 1

b[i:, :(N-j)] = a[i:, :(N-j)]
