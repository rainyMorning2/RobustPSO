import numpy as np


class f3(object):
    low = np.array([0, -30])
    high = np.array([1, 30])
    name = 'ZDT4'
    def __init__(self, N, D=2, K=2):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = position[i][0]
            g = 11+position[i][1]**2 - 10*np.cos(2*np.pi*position[i][1])
            if position[i][0] <= g:
                result[i][1] = g * (1 - (position[i][0]/g)**0.5)
            else:
                result[i][1] = 0

        return result, cnt


class f2(object):
    low = np.array([-5, -5, -5])
    high = np.array([5, 5, 5])
    name = 'Kursawe'
    def __init__(self, N, D=3, K=2):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = -10 * np.e ** (-0.2 * (position[i][0] ** 2 + position[i][1] ** 2) ** 0.5) \
                           - 10 * np.e ** (-0.2 * (position[i][1] ** 2 + position[i][2] ** 2) ** 0.5)
            for j in range(0, self.D):
                result[i][1] += abs(position[i][j]) ** 0.8 + 5 * np.sin(position[i][j] ** 3)

        return result, cnt


class f1(object):
    low = np.array([0, 0])
    high = np.array([7, 7])
    name = "Kita"

    def __init__(self, N, D=2, K=2):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = position[i][0] ** 2 - position[i][1]
            result[i][1] = -1 / 2 * position[i][0] - position[i][1] - 1

            if ~(1.0 / 6 * position[i][0] + position[i][1] - 13.0 / 2 <= 0):
                cnt[i] += 1
            if ~(1.0 / 2 * position[i][0] + position[i][1] - 15.0 / 2 <= 0):
                cnt[i] += 1
            if ~(5 * position[i][0] + position[i][1] - 30 <= 0):
                cnt[i] += 1

        return result, cnt
