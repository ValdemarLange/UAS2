import matplotlib.pyplot as plt
import numpy as np

def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x**2) ** 2

delta = 0.01
x = np.arange(-4.0, 4.001, delta)
y = np.arange(-3.0, 3.001, delta)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
levels = 0.1 * 2 ** np.arange(0, 18, 1)
plt.contour(X, Y, Z, levels)
plt.show()