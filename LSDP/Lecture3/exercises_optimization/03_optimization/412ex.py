import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10.0, 10.0, 0.1)
y = np.sin(x)
gy = x - 0.5 * np.cos(x)
plt.plot(x, y, label='f(x)')
plt.plot(x, gy, label='g(x)')
plt.show()