import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10.0, 10.0, 0.01)
y = np.cos(2*x)
gy = x - 0.5 * np.diff(y, append=0)
plt.plot(x, y, label='f(x)')
plt.plot(x, gy, label='g(x)')
plt.show()