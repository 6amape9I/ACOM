import numpy as np


def fill_matrix(core_size, standard_deviation):
    core = np.ones((core_size, core_size))
    a = b = (core_size + 1) // 2

    for i in range(core_size):
        for j in range(core_size):
            core[i, j] = gauss(i, j, standard_deviation, a, b)

    return core


def gauss(x, y, omega, a, b):
    omega2 = 2 * omega ** 2

    m1 = 1 / (np.pi * omega2)
    m2 = np.exp(-((x - a) ** 2 + (y - b) ** 2) / omega2)

    return m1 * m2


if __name__ == '__main__':

    core_sizes = [3, 5, 7]
    standard_deviation = 10

    for core_size in core_sizes:
        core = fill_matrix(core_size, standard_deviation)
        print(core)
