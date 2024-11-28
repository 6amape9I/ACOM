import numpy as np
from Laba_3.fill_matrix import fill_matrix


def norm_matrix(core_size, core):
    sum = 0
    for i in range(core_size):
        for j in range(core_size):
            sum += core[i, j]

    for i in range(core_size):
        for j in range(core_size):
            core[i, j] /= sum

    return core


if __name__ == '__main__':

    core_sizes = [3, 5, 7]
    standard_deviation = 10

    for core_size in core_sizes:
        core = fill_matrix(core_size, standard_deviation)
        print(core)
        core = norm_matrix(core_size, core)
        print(core)

        #Проверка
        sum = 0
        for i in core:
            for j in i:
                sum += j
        print(sum)

