import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10.0, 10.0, 0.01)
y = np.sin(x)

dy = np.gradient(y, 0.01)
ddy = np.gradient(dy, 0.01)
k= 0.9
g = x - dy / (ddy+k)
plt.plot(x, y, label='f(x)')
plt.plot(x, g, label='g(x)')
plt.show()