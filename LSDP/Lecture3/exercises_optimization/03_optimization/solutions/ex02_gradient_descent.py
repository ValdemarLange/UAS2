import matplotlib.pyplot as plt
import numpy as np


def main():
    # Data for plotting
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.sin(x)
    dy = np.cos(x)
    xcleaned = x[:-1]
    yiterated = x - 0.5 * dy

    plt.plot(x, y)
    plt.plot(xcleaned, xcleaned, ":k")
    plt.plot(x, yiterated, "r")

    plt.show()


main()
