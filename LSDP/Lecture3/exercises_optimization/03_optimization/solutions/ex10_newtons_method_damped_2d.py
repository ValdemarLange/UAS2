import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from matplotlib.widgets import Button, Slider


def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x**2) ** 2


delta = 0.005
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

    ddfdydy = (
        rosenbrock(x, y + eps) - 2 * rosenbrock(x, y) + rosenbrock(x, y - eps)
    ) / eps**2
    ddfdxdx = (
        rosenbrock(x + eps, y) - 2 * rosenbrock(x, y) + rosenbrock(x - eps, y)
    ) / eps**2
    ddfdydx = (
        rosenbrock(x + eps, y + eps)
        - rosenbrock(x + eps, y)
        - rosenbrock(x, y + eps)
        + rosenbrock(x, y)
    ) / eps**2
    ddfdxdy = (
        rosenbrock(x + eps, y + eps)
        - rosenbrock(x + eps, y)
        - rosenbrock(x, y + eps)
        + rosenbrock(x, y)
    ) / eps**2

    gradient = np.array([[dfdx], [dfdy]])
    hessian = np.array([[ddfdxdx, ddfdydx], [ddfdxdy, ddfdydy]])
    update_step = -np.linalg.inv(hessian + scale * np.eye(2)) @ gradient
    xp = x + update_step[0, 0]
    yp = y + update_step[1, 0]
    return np.array([xp, yp])


points = np.array([[4, 2.0], [1, 3], [-3, 3]])
(points_plot_element,) = ax.plot(points[:, 0], points[:, 1], "ro")


fig.subplots_adjust(left=0.25)
axslider = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
scale_slider = Slider(
    ax=axslider,
    label="scale",
    valmin=-10,
    valmax=10,
    valinit=10,
    orientation="vertical",
)


def slider_changed(val):
    damping = np.exp(scale_slider.val)
    scale_slider.valtext.set_text(f"{damping:.2f}")


slider_changed(0)

scale_slider.on_changed(slider_changed)
axbutton = fig.add_axes([0.1, 0.1, 0.10, 0.05])
iterate_button = Button(ax=axbutton, label="Iterate")


def iterate(val):
    ic(points)
    damping = np.exp(scale_slider.val)
    for idx, point in enumerate(points):
        point = update_rule(*point, damping)
        points[idx] = point
    ic(points)
    points_plot_element.set_xdata(points[:, 0])
    points_plot_element.set_ydata(points[:, 1])
    fig.canvas.draw_idle()


iterate_button.on_clicked(iterate)

plt.show()
