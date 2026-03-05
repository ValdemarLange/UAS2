import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10.0, 10.0, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()