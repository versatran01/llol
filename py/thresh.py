import numpy as np

phi = np.pi / 2.5
dtheta = np.pi * 2 / 1024
n = 5
theta = n * dtheta

tr = 1 / (np.cos(theta) - np.sin(theta) * np.tan(phi)) + 0.02
tr2 = 1 / (1 - theta * np.tan(phi)) + 0.02
print(tr, tr2)