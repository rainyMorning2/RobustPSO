import numpy as np
import copy
import math


class testRobust(object):
    low = -1
    high = 4.5

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        result = np.zeros(position.shape[0])

        for j in range(0, position.shape[0]):
            result[j] += 2 * position[j][0] ** 6 - 12.2 * position[j][0] ** 5 + 21.2 * position[j][0] ** 4 \
                         - 6.4 * position[j][0] ** 3 - 4.7 * position[j][0] ** 2 + 6.2 * position[j][0] \
                         + position[j][1] ** 6 - 11 * position[j][1] ** 5 + 43.3 * position[j][1] ** 4 \
                         - 74.8 * position[j][1] ** 3 + 56.9 * position[j][1] ** 2 - 10 * position[j][1] \
                         - 0.1 * (position[j][1] ** 2) * (position[j][0] ** 2) - 4.1 * position[j][1] * position[j][0] \
                         + 0.4 * position[j][0] * position[j][1] ** 2 + 0.4 * position[j][1] * position[j][0] ** 2
        return result
