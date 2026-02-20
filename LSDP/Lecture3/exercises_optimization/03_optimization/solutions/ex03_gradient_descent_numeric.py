import matplotlib.pyplot as plt
import numpy as np


def main():
    # Data for plotting
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.exp(x) + np.exp(-x)
    y = np.sin(x)
    dy = np.diff(y, append=0) / np.diff(x, append=1)
    yiterated = x - 1 * dy

    xcleaned = x[:-1]
    ycleaned = y[:-1]
    yiteratedcleaned = yiterated[:-1]

    plt.plot(xcleaned, ycleaned)
    plt.plot(xcleaned, xcleaned, ":k")
    plt.plot(xcleaned, yiteratedcleaned, "r")

    plt.show()


main()
