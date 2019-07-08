import numpy as np
import copy
import math


class ShiftedSphere(object):
    low = -90
    high = 110
    goal = 0.01

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)

        for j in range(0, self.N):
            for i in range(0, self.D):
                result[j] += (position[j][i] - 10) ** 2

        return result


class ShiftedAckley(object):
    low = -27
    high = 37
    goal = 0.01

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)

        for j in range(0, self.N):
            a = 0.0
            b = 0.0
            for i in range(0, self.D):
                a += (position[j][i] - 5) ** 2
                b += math.cos(2 * math.pi * (position[j][i] - 5))
            result[j] = -20 * math.exp(-0.2 * ((a / self.D) ** 0.5)) - math.exp(b / self.D) + 20 + math.e
        return result


class ShiftedRosenbrock(object):
    low = -1.048
    high = 3.048
    goal = 100

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)

        for j in range(0, self.N):
            for i in range(0, self.D - 1):
                result[j] += 100 * (position[j][i + 1] - 1 - (position[j][i] - 1) ** 2) ** 2 + (
                            position[j][i] - 1 - 1) ** 2

        return result


class ShiftedRastrigin(object):
    low = -2.12
    high = 8.12
    goal = 100

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)
        for j in range(0, self.N):
            for i in range(0, self.D):
                result[j] += (position[j][i] - 3) ** 2 + (- 10 * math.cos(2 * np.pi * (position[j][i] - 3)) + 10)
        return result


class Step(object):
    low = -100
    high = 100

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)
        for j in range(0, self.N):
            for i in range(0, self.D):
                result[j] += (np.floor(position[j][i] + 0.5)) ** 2
        return result


class Schwefel_1_2(object):
    low = -100
    high = 100
    goal = 0.01

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)
        for j in range(0, self.N):
            for i in range(0, self.D):
                a = 0.0
                for k in range(0, i):
                    a += position[j][k]
                result[j] += a ** 2
        return result


class RotatedGriewank(object):
    low = -600
    high = 600
    goal = 0.05

    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.M = np.linalg.qr(np.random.rand(D, D))[0]

    def getValue(self, position):
        new = copy.deepcopy(position)
        for i in range(0, self.N):
            new[i] = self.M.dot(new[i])
        return Griewank(self.N, self.D).getValue(new)


class RotatedWeierstrass(object):
    low = -0.5
    high = 0.5
    goal = 1

    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.M = np.linalg.qr(np.random.rand(D, D))[0]

    def getValue(self, position):
        new = copy.deepcopy(position)
        for i in range(0, self.N):
            new[i] = self.M.dot(new[i])
        return Weierstrass(self.N, self.D).getValue(new)


class RotatedNoncontinuousRastrigin(object):
    low = -5.12
    high = 5.12
    goal = 100

    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.M = np.linalg.qr(np.random.rand(D, D))[0]

    def getValue(self, position):
        new = copy.deepcopy(position)
        for i in range(0, self.N):
            new[i] = self.M.dot(new[i])
        return NoncontinuousRastrigin(self.N, self.D).getValue(new)


class RotatedRastrigin(object):
    low = -5.12
    high = 5.12
    goal = 100

    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.M = np.linalg.qr(np.random.rand(D, D))[0]

    def getValue(self, position):
        new = copy.deepcopy(position)
        for i in range(0, self.N):
            new[i] = self.M.dot(new[i])
        return Rastrigin(self.N, self.D).getValue(new)


class RotatedSchwefel(object):
    low = -500
    high = 500
    goal = 2000

    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.M = np.linalg.qr(np.random.rand(D, D))[0]

    def getValue(self, position):
        new = copy.deepcopy(position)
        for i in range(0, self.N):
            new[i] = self.M.dot(new[i] - 420.96) + 420.96
            for j in range(0, self.D):
                if abs(new[i][j]) <= 500:
                    new[i][j] = new[i][j] * np.sin(np.sqrt(abs(new[i][j])))
                else:
                    new[i][j] = 0.001 * (abs(new[i][j]) - 500) ** 2

        return Schwefel(self.N, self.D).getValue(new)


class RotatedAckley(object):
    low = -32
    high = 32
    goal = 0.01

    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.M = np.linalg.qr(np.random.rand(D, D))[0]

    def getValue(self, position):
        new = copy.deepcopy(position)
        for i in range(0, self.N):
            new[i] = self.M.dot(new[i])
        return Ackley(self.N, self.D).getValue(new)


class Griewank(object):
    low = -600
    high = 600
    goal = 0.05

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)
        for j in range(0, self.N):
            a = 0.0
            b = 1.0
            for i in range(0, self.D):
                a += position[j][i] ** 2
                b *= math.cos(position[j][i] / np.sqrt(i + 1))
            result[j] = a / 4000 + (1 - b)
        return result


class Weierstrass(object):
    low = -0.5
    high = 0.5
    goal = 0.01

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        km = 20
        a = 0.5
        b = 3

        result = np.zeros(self.N)
        for j in range(0, self.N):
            for i in range(0, self.D):
                part1 = 0
                part2 = 0
                for k in range(0, km):
                    part1 += (a ** k) * np.cos(2 * np.pi * (b ** k) * (position[j][i] + 0.5))
                    part2 += a ** k * np.cos(np.pi * (b ** k))
                result[j] += part1 - part2
        return result


class NoncontinuousRastrigin(object):
    low = -5.12
    high = 5.12
    goal = 100

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        new = copy.deepcopy(position)
        for i in range(0, self.N):
            for j in range(0, self.D):
                if abs(new[i][j]) > 0.5:
                    new[i][j] = round(2 * new[i][j]) / 2
        return Rastrigin(self.N, self.D).getValue(new)


class Rastrigin(object):
    low = -5.12
    high = 5.12
    goal = 100

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)
        for j in range(0, self.N):
            for i in range(0, self.D):
                result[j] += (10 - 10 * np.cos(2 * np.pi * position[j][i])) + position[j][i] ** 2
        return result


class Schwefel(object):
    low = -500
    high = 500
    goal = 2000

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)
        for j in range(0, self.N):
            for i in range(0, self.D):
                result[j] += (418.9829 - position[j][i] * math.sin(np.sqrt(np.abs(position[j][i]))))
        return result


class Ackley(object):
    low = -32
    high = 32
    goal = 0.01

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)

        for j in range(0, self.N):
            a = 0.0
            b = 0.0
            for i in range(0, self.D):
                a += position[j][i] ** 2
                b += math.cos(2 * math.pi * position[j][i])
            result[j] = -20 * math.exp(-0.2 * ((a / self.D) ** 0.5)) - math.exp(b / self.D) + 20 + math.e
        return result


class Noise(object):
    low = -1.28
    high = 1.28
    goal = 0.05

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):
        result = np.zeros(self.N)
        for j in range(0, self.N):
            for i in range(0, self.D):
                result[j] += (i + 1) * position[j][i] ** 4
            result[j] += np.random.rand(1)
        return result


class Schwefel_2_22(object):
    low = -10
    high = 10
    goal = 0.01

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)
        for j in range(0, self.N):
            a = 0.0
            b = 1.0
            for i in range(0, self.D):
                a += abs(position[j][i])
                b *= abs(position[j][i])
            result[j] = a + b
        return result


class Rosenbrock(object):
    low = -2.048
    high = 2.048
    goal = 100

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)

        for j in range(0, self.N):
            for i in range(0, self.D - 1):
                result[j] += 100 * (position[j][i + 1] - position[j][i] ** 2) ** 2 + (position[j][i] - 1) ** 2

        return result


class Sphere(object):
    low = -100
    high = 100
    goal = 0.01

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def getValue(self, position):

        result = np.zeros(self.N)

        for j in range(0, self.N):
            for i in range(0, self.D):
                result[j] += position[j][i] ** 2

        return result
