import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np
from icecream import ic


def main():
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.exp(x) + np.exp(-x) - 2
    y = np.sin(x)
    dy = np.diff(y, append=0) / np.diff(x, append=1)
    ddy = np.diff(dy, append=0) / np.diff(x, append=1)
    yiterated = x - dy / (ddy + 0)

    xcleaned = x[:-2]
    ycleaned = y[:-2]
    yiteratedcleaned = yiterated[:-2]

    fig, ax = plt.subplots()
    ax.plot(xcleaned, ycleaned, linewidth=3)
    ax.plot(xcleaned, xcleaned, ":k")
    (damped_plot,) = ax.plot(xcleaned, yiteratedcleaned, "r")

    fig.subplots_adjust(left=0.25)

    axdamping = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    damping_slider = Slider(
        ax=axdamping,
        label="Damping",
        valmin=-1.0,
        valmax=2.0,
        valinit=0.0,
        orientation="vertical",
    )

    def update(val):
        yiterated = x - dy / (ddy + damping_slider.val)
        yiteratedcleaned = yiterated[:-2]
        damped_plot.set_ydata(yiteratedcleaned)
        fig.canvas.draw_idle()

    damping_slider.on_changed(update)

    ax.legend(("f(x)", "y = x", "yiterated", "yiterateddamped"), loc="lower right")

    ax.set_ylim(-10, 10)
    plt.show()


main()
