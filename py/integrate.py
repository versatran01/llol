import numpy as np
import matplotlib.pyplot as plt

n = 10
x0 = np.zeros((10), float)
x1 = np.zeros((10), float)

a0 = 0.2
a1 = a0 + 0.1
dt = 0.01

v0 = 0
v1 = 0
for i in range(1, 10):
    
    x0[i] = x0[i-1] + v0 * dt + 0.5 * a0 * dt * dt
    v0 = v0 + a0 * dt
    
    x1[i] = x1[i-1] + v1 * dt + 0.5 * a1 * dt * dt
    v1 = v1 + a1 * dt

f, ax = plt.subplots()
ax.plot(x0)
ax.plot(x1)
ax.plot(x1 - x0)