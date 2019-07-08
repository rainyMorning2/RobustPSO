import numpy as np


# cons
class Kita(object):
    low = np.array([0, 0])
    high = np.array([7, 7])
    name = "Kita"

    def __init__(self, N, D=2, K=2, C=3):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains which means the max value is C(begin from 0)

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


class Binh2(object):
    low = np.array([-15, -15])
    high = np.array([30, 30])
    name = 'Binh2'

    def __init__(self, N, D=2, K=2, C=2):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = 4 * position[i][0]**2 + 4*position[i][1]**2
            result[i][1] = (position[i][0]-5)**2 + (position[i][1]-5)**2

            if ~((position[i][0]-5)**2 + position[i][1]**2 - 25 <= 0):
                cnt[i] += 1
            if ~(-(position[i][0]-8)**2 - (position[i][1]-3)**2 + 7.7 <= 0):
                cnt[i] += 1

        return result, cnt


class Belegundu(object):
    low = np.array([0, 0])
    high = np.array([5, 3])
    name = 'Belegundu'

    def __init__(self, N, D=2, K=2, C=2):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = -2 * position[i][0] + position[i][1]
            result[i][1] = 2 * position[i][0] + position[i][1]

            if ~(-position[i][0] + position[i][1] - 1 <= 0):
                cnt[i] += 1
            if ~(position[i][0] + position[i][1] - 7 <= 0):
                cnt[i] += 1

        return result, cnt


class DTLZ8(object):
    low = np.array([0, 0]*15)
    high = np.array([1, 1]*15)
    name = 'DTLZ8'

    def __init__(self, N, D=30, K=3, C=3):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = 0.1*sum(position[i][0:10])
            result[i][1] = 0.1*sum(position[i][10:20])
            result[i][2] = 0.1*sum(position[i][20:30])

            if ~(result[i][2] + 4 * result[i][0] - 1 >= 0):
                cnt[i] += 1
            if ~(result[i][2] + 4 * result[i][1] - 1 >= 0):
                cnt[i] += 1
            if ~(2*result[i][2] + result[i][1] + result[i][0] - 1 >= 0):
                cnt[i] += 1

        return result, cnt


class Tanaka(object):
    low = np.array([0, 0])
    high = np.array([np.pi, np.pi])
    name = 'Tanaka'

    def __init__(self, N, D=2, K=2, C=2):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = position[i][0]
            result[i][1] = position[i][1]

            if ~(position[i][0]**2 + position[i][1]**2 - 1 - 0.1*np.cos(16*np.arctan(position[i][0]/position[i][1])) >= 0):
                cnt[i] += 1
            if ~((position[i][0]-0.5)**2 + (position[i][1]-0.5)**2 - 0.5 <= 0):
                cnt[i] += 1

        return result, cnt


# nonCons
class ZDT4(object):
    low = np.array([0, -30])
    high = np.array([1, 30])
    name = 'ZDT4'

    def __init__(self, N, D=2, K=2,C=0):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = position[i][0]
            g = 11 - 10*np.cos(2*np.pi*position[i][1]) + position[i][1]**2
            if position[i][0] <= g:
                result[i][1] = g * (1 - (position[i][0]/g)**0.5)
            else:
                result[i][1] = 0
        return result, cnt


class Kursawe(object):
    low = np.array([-5, -5, -5])
    high = np.array([5, 5, 5])
    name = 'Kursawe'

    def __init__(self, N, D=3, K=2, C=0):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = -10 * np.e ** (-0.2 * (position[i][0] ** 2 + position[i][1] ** 2) ** 0.5) \
                           - 10 * np.e ** (-0.2 * (position[i][1] ** 2 + position[i][2] ** 2) ** 0.5)
            for j in range(0, self.D):
                result[i][1] += abs(position[i][j]) ** 0.8 + 5 * np.sin(position[i][j] ** 3)

        return result, cnt


class ZDT3(object):
    low = np.array([0, 0])
    high = np.array([1, 1])
    name = 'ZDT3'

    def __init__(self, N, D=2, K=2,C=0):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):

            result[i][0] = position[i][0]
            g = 1 + 9 * position[i][1]
            result[i][1] = g * (1 - (position[i][0]/g)**0.5 - position[i][0]/g*np.sin(10*np.pi*position[i][0]))

        return result, cnt


class DTLZ1(object):
    low = np.array([0, 0, 0])
    high = np.array([1, 1, 1])
    name = 'DTLZ1'

    def __init__(self, N, D=3, K=3,C=0):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):
            g = 100*(1+(position[i][2]-0.5)**2 - np.cos(20*np.pi*(position[i][2]-0.5)))
            result[i][0] = 0.5 * position[i][0] * position[i][1] * (1+g)
            result[i][1] = 0.5 * position[i][0] * (1-position[i][1]) * (1+g)
            result[i][2] = 0.5 * (1-position[i][0]) * (1+g)

        return result, cnt


class DTLZ2(object):
    low = np.array([0, 0]*6)
    high = np.array([1, 1]*6)
    name = 'DTLZ2'

    def __init__(self, N, D=12, K=3,C=0):
        self.N = N  # number of particles
        self.D = D  # dimensions of vector x
        self.K = K  # dimensions of vector f(x)
        self.C = C  # number of constains

    def getValue(self, position):

        result = np.zeros([self.N, self.K])
        cnt = np.zeros(self.N)

        for i in range(0, self.N):
            g = 0
            for j in range(2, self.D):
                g += (position[i][j]-0.5)**2

            result[i][0] = np.cos(0.5*np.pi*position[i][0]) * np.cos(0.5*np.pi*position[i][1]) * (1+g)
            result[i][1] = np.cos(0.5*np.pi*position[i][0]) * np.sin(0.5*np.pi*position[i][1]) * (1+g)
            result[i][2] = np.sin(0.5*np.pi*position[i][0]) * (1+g)

        return result, cnt
