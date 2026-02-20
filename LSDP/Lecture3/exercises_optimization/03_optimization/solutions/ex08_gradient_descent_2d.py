import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from matplotlib.widgets import Button, Slider


def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x**2) ** 2


delta = 0.01
x = np.arange(-4.0, 4.001, delta)
y = np.arange(-3.0, 3.001, delta)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

fig, ax = plt.subplots()

levels = 0.1 * 2 ** np.arange(0, 18, 1)
ax.contour(X, Y, Z, levels)


ax.axis("equal")


def update_rule(x, y, scale=0.01):
    eps = 0.000001
    dfdx = (rosenbrock(x + eps, y) - rosenbrock(x, y)) / eps
    dfdy = (rosenbrock(x, y + eps) - rosenbrock(x, y)) / eps
    gradient = np.array([dfdx, dfdy])
    gradient_normed = gradient / np.linalg.norm(gradient)
    xp = x - scale * gradient_normed[0]
    yp = y - scale * gradient_normed[1]
    return np.array([xp, yp])


points = np.array([[4, 2.0]])
(points_plot_element,) = ax.plot(points[:, 0], points[:, 1], "ko")


fig.subplots_adjust(left=0.25)
axslider = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
scale_slider = Slider(
    ax=axslider, label="scale", valmin=0, valmax=1, valinit=0.1, orientation="vertical"
)
scale_slider.on_changed(lambda b: None)
axbutton = fig.add_axes([0.1, 0.1, 0.10, 0.05])
iterate_button = Button(ax=axbutton, label="Iterate")


def iterate(val):
    print("iterating")
    ic(points)
    for idx, point in enumerate(points):
        point = update_rule(*point, scale_slider.val)
        ic(point)
        points[idx] = point
    points_plot_element.set_xdata(points[:, 0])
    points_plot_element.set_ydata(points[:, 1])
    fig.canvas.draw_idle()


iterate_button.on_clicked(iterate)

plt.show()
