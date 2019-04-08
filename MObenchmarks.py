import numpy as np


class f1(object):
    low = np.array([0, 0])
    high = np.array([7, 7])

    def __init__(self, N, D=2, K=2):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = -(-position[i][0] ** 2 + position[i][1])
            result[i][1] = -(1.0 / 2 * position[i][0] + position[i][1] + 1)

            if ~(1.0 / 6 * position[i][0] + position[i][1] - 13.0 / 2 <= 0):
                cnt[i] += 1
            if ~(1.0 / 2 * position[i][0] + position[i][1] - 15.0 / 2 <= 0):
                cnt[i] += 1
            if ~(5 * position[i][0] + position[i][1] - 30 <= 0):
                cnt[i] += 1

        return result, cnt
