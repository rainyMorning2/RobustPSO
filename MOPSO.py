import numpy as np


def checkBoundary(position, velocity, low, high):
    for i in range(0, position.shape[0]):
        for j in range(0, position.shape[1]):
            if position[i][j] < low[i]:
                position[i][j] = low[i]
                velocity[i] *= -1
            if position[i][j] > high[i]:
                position[i][j] = high[i]
                velocity[i] *= -1


def isDominated(value1, value2):
    # return whether value1 dominates value2 or not

    flag = False
    for x, y in zip(value1, value2):
        if x > y:
            return False
        elif x < y:
            flag = True

    if flag:
        return True
    else:
        return False


def MOPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    K = benchmark.K
    w = 0.4
    mutrate = 0.5

    repsize = 100
    divisions = 30

    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])

    pb = position
    storedFit, storedCons = benchmark.getValue(pb)

    t = 0
    while t < T:

        # check boundary 1
        checkBoundary(position, velocity, benchmark.low, benchmark.high)

        # mutation 1
        if (1 - 1.0 * t / T) ** (5.0 / mutrate) != 0:
            for i in range(0, N):
                dim = np.random.randint(0, D)
                mutrange = (benchmark.high[dim] - benchmark.low[dim]) * (1 - 1.0 * t / T) ** (5.0 / mutrate)
                hb = position[i][dim] + mutrange
                lb = position[i][dim] - mutrange

                if hb > benchmark.high[dim]:
                    hb = benchmark.high[dim]
                if lb < benchmark.low[dim]:
                    lb = benchmark.low[dim]
                position[i][dim] = np.random.rand() * (hb - lb) + lb

        # evaluate particles 1
        currentFit, currentCons = benchmark.getValue(position)

        # update repo

        # update pb 1
        for i in range(0, N):
            if currentCons[i] < storedCons[i]:
                storedCons[i] = currentCons[i]
                storedFit[i] = currentFit[i]
                pb = position[i]
            elif currentCons[i] == storedCons[i]:
                if isDominated(currentFit[i], storedFit[i]):
                    storedCons[i] = currentCons[i]
                    storedFit[i] = currentFit[i]
                    pb = position[i]
                elif isDominated(storedFit[i], currentFit[i]):
                    pass
                else:
                    if np.random.rand() > 0.5:
                        storedCons[i] = currentCons[i]
                        storedFit[i] = currentFit[i]
                        pb = position[i]

        # update vel
        velocity += w * velocity + np.random.rand(N, D) * (pb - position) + np.random.rand(N, D) * (repo - position)

        # update position 1
        position += velocity

        t += 1


def main():
    pass


if __name__ == '__main__':
    low = [0, 0]
    high = [1, 30]
    v=[[[] for i in range(3)] for j in range(3)]
    print(v)
    print(v[0][1].append(2))
    print(v[0][1].append(23))
    print(v)