import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def main():
    # Data for plotting
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.exp(x) + np.exp(-x) - 1.9
    y = np.sin(x)
    dy = np.diff(y, append=0) / np.diff(x, append=1)
    yiterated = x - 0 * dy

    xcleaned = x[:-1]
    ycleaned = y[:-1]
    yiteratedcleaned = yiterated[:-1]

    fig, ax = plt.subplots()
    ax.plot(xcleaned, ycleaned, linewidth=3)
    ax.plot(xcleaned, xcleaned, ":k")
    (iterated_plot,) = ax.plot(xcleaned, yiteratedcleaned, "r")
    ax.set_ylim(-10, 10)

    fig.subplots_adjust(left=0.25)
    axdamping = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    damping_slider = Slider(
        ax=axdamping,
        label="k",
        valmin=-5.0,
        valmax=5.0,
        valinit=0.0,
        orientation="vertical",
    )

    def update(val):
        yiterated = x - damping_slider.val * dy
        yiteratedcleaned = yiterated[:-1]
        iterated_plot.set_ydata(yiteratedcleaned)
        fig.canvas.draw_idle()

    damping_slider.on_changed(update)

    plt.show()


main()
