import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10.0, 10.0, 0.01)
y = np.sin(x)
y = np.exp(x) + np.exp(-x)
k = 50
# k = 1
dy = np.gradient(y, 0.01)
gy = x - k * dy #np.diff(y, append=0)
plt.plot(x, y, label='f(x)')
plt.plot(x, gy, label='g(x)')
plt.show()