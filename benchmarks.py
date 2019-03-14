import numpy as np
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


class f1(object):
    low = -2
    high = 2

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        result = np.zeros(position.shape[0])

        for j in range(0, position.shape[0]):
            for i in range(0, self.D):
                if position[j][i] < 0:
                    result[j] += -(position[j][i] + 1) ** 2 + 1.4 - 0.8 * abs(math.sin(6.283 * position[j][i]))
                else:
                    result[j] += 0.6 * 2 ** (-8 * abs(position[j][i] - 1)) + 0.958887 - 0.8 * abs(
                        math.sin(6.283 * position[j][i]))
        return result


class f2(object):
    low = -0.5
    high = 0.5

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        result = np.zeros(position.shape[0])

        for j in range(0, position.shape[0]):
            for i in range(0, self.D):
                if -0.2 <= position[j][i] < 0.2:
                    result[j] -= 1

        return result


class f3(object):
    low = -1
    high = 1

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        result = np.zeros(position.shape[0])

        for j in range(0, position.shape[0]):
            for i in range(0, self.D):
                if -0.8 <= position[j][i] < 0.2:
                    result[j] -= position[j][i] + 0.8

        return result


class f5(object):
    low = -1.5
    high = 1.5

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        result = np.zeros(position.shape[0])

        for j in range(0, position.shape[0]):
            for i in range(0, self.D):
                if -0.6 <= position[j][i] < -0.2:
                    result[j] -= 1
                elif 0.2 <= position[j][i] < 0.36:
                    result[j] -= 1.25
                elif 0.44 <= position[j][i] < 0.6:
                    result[j] -= 1.25

        return result


class f6(object):
    low = -2 * math.pi
    high = 4 * math.pi

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        result = np.zeros(position.shape[0])

        for j in range(0, position.shape[0]):
            for i in range(0, self.D):
                if self.low <= position[j][i] < 2 * math.pi:
                    result[j] -= math.cos(0.5 * position[j][i]) + 1
                elif 2 * math.pi <= position[j][i] < 4 * math.pi:
                    result[j] -= 1.1 * math.cos(position[j][i] + math.pi) + 1.1

        return result
