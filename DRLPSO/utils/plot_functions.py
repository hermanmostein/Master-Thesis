import matplotlib.pyplot as plt
from functions import rastrigrin, zakharov, rosenbrock, schwefel
import numpy as np


def plot_function(f, n_points=100):

    bounds = f(None)
    x_bottom, x_top = bounds[0]
    y_bottom, y_top = bounds[1]

    ax = plt.axes(projection='3d')
    X = np.linspace(x_bottom, x_top, n_points)
    Y = np.linspace(y_bottom, y_top, n_points)

    Z = np.zeros((n_points, n_points))
    for i in range(len(X)):
        for j in range(len(Y)):
            Z[j, i] = f(np.array([X[i], Y[j]]))

    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, Z, cmap='winter')
    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.set_zlabel("Z label")
    plt.show()


plot_function(rastrigrin)
