import matplotlib.pyplot as plt
import numpy as np
from icecream import ic


def main():
    # Data for plotting
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.exp(x) + np.exp(-x) - 2
    y = np.sin(x)
    dy = np.diff(y, append=0) / np.diff(x, append=1)
    ddy = np.diff(dy, append=0) / np.diff(x, append=1)
    yiterated = x - dy / ddy
    x = x[:-2]
    y = y[:-2]
    dy = dy[:-2]
    ddy = ddy[:-2]
    yiterated = yiterated[:-2]

    plt.plot(x, y, linewidth=3)
    plt.plot(x, x, ":k")
    plt.plot(x, yiterated, "r")
    plt.plot(x, dy, "g")
    plt.plot(x, ddy, "b")

    plt.legend(("f(x)", "y = x", "yiterated", "dy", "ddy"), loc="lower right")

    plt.ylim(-100, 100)

    plt.show()


main()
