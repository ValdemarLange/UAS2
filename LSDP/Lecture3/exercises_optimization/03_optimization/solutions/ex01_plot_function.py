import matplotlib.pyplot as plt
import numpy as np


def main():
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()


main()
