import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
from numba import njit
from itertools import product

global seed
seed = random.random()


def simple_test(coo):
    if (coo is None):
        return np.ones((2, 2))*[-10, 10]
    return np.sum(np.square(coo))

# Training


def zakharov(coo):
    # minimum = [0, ..., 0]
    if (coo is None):
        return np.ones((20, 2))*[-5, 10]

    idx = np.arange(1, len(coo)+1)
    first = np.sum(np.square(coo))
    second = np.square(0.5*np.dot(idx, coo))
    third = (0.5*np.dot(idx, coo))**4
    return first + second + third


def rastrigrin(coo, size=20):
    # minimum = [0, ..., 0]
    if (coo is None):
        return np.ones((20, 2))*[-5.12, 5.12]
    A = 10
    temp = np.sum(np.square(coo) - A*np.cos(2*np.pi*coo))
    return A*len(coo) + temp


def rosenbrock(coo):
    # minimum = [1, ..., 1]
    if (coo is None):
        return np.ones((20, 2))*[-10, 10]
    forward = coo[1:]
    back = coo[:-1]
    return np.sum(100*np.square(forward - np.square(back)) + np.square(back-1))


def schwefel(coo):
    # minimum = [420,96874636, ...]
    if (coo is None):
        return np.ones((20, 2))*[-500, 500]
    d = len(coo)
    s = np.sum(coo*np.sin(np.sqrt(np.absolute(coo))))
    return 418.9829*d - s


def ackley(coo):
    # minimum = [0, ..., 0]
    if (coo is None):
        return np.ones((20, 2))*[-15, 30]
    N = len(coo)
    first = -0.2*np.sqrt((1/N) * np.sum(np.square(coo)))
    second = (1/N) * np.sum(np.cos(2*np.pi*coo))
    return 20-20*np.exp(first) + np.e - np.exp(second)


def translated_rastrigin(r=0.5):
    # minimum = [-r, ..., -r]
    def f(coo):
        if (coo is None):
            return np.ones((2, 2))*[-5.12, 5.12]
        fill = 5*r - 2
        c = np.full_like(coo, fill, dtype=np.double)
        A = 10
        temp = np.sum(np.square(coo+c) - A*np.cos(2*np.pi*(coo+c)))
        return A*len(coo) + temp
    return f


def translated_zakharov(coo):
    # minimum = [8, ..., 8]
    if (coo is None):
        return np.ones((20, 2))*[-5, 10]
    coo -= 8
    return zakharov(coo)


def translated_ackley(coo):
    # minimum = [-r, ..., -r]
    if (coo is None):
        return np.ones((20, 2))*[-15, 30]

    r = 10*random.random() - 5
    c = np.full_like(coo, r, dtype=np.double)
    N = len(coo)
    first = -0.2*np.sqrt((1/N) * np.sum(np.square(coo-c)))
    second = (1/N) * np.sum(np.cos(2*np.pi*(coo-c)))
    return 20-20*np.exp(first) + np.e - np.exp(second)


# Testing


def griewangk(coo):
    # minimum = [0, ..., 0]
    if (coo is None):
        return np.ones((20, 2))*[-600, 600]
    coo = np.array(coo)
    s = np.sum(np.square(coo))
    sq = np.sqrt(np.arange(1, coo.shape[0]+1))
    temp = coo/sq
    p = np.product(np.cos(temp))

    return 1 + s/4000 - p


# Helpers

def plot_function(f):
    min_value = np.infty
    min_pos = None
    lims = f(None)[0]
    n = 300
    map = np.zeros((n, n))
    X = np.linspace(int(lims[0]), int(lims[1]), n)
    Y = np.linspace(int(lims[0]), int(lims[1]), n)
    X_ = np.linspace(int(lims[0]), int(lims[1]), n//10)
    Y_ = np.linspace(int(lims[0]), int(lims[1]), n//10)
    for i in range(n):
        for j in range(n):
            v = f(np.array([X[i], Y[j]]))
            map[i, j] = v
            if (v < min_value):
                min_value = v
                min_pos = [X[i], Y[j]]

    print(min_value, min_pos)

    plt.title(f.__name__)
    plt.imshow(map)
    plt.colorbar()
    plt.show()
