import numpy as np
import networkx as nx


def PSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c1 = 1.494
    c2 = 1.494
    w = 0.729

    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    fitness = np.zeros(N)
    lastFitness = np.zeros(N) + 999999999999999
    bestFitness = 999999999999999
    Pb = np.zeros([N, D])
    Gb = np.zeros(D)
    fitData = []

    t = 0
    while t < T:

        if np.amin(position) < benchmark.low or np.amax(position) > benchmark.high:
            for i in range(0, N):
                for j in range(0, D):
                    while position[i][j] < benchmark.low:
                        position[i][j] = benchmark.low
                    while position[i][j] > benchmark.high:
                        position[i][j] = benchmark.high

        fitness = benchmark.getValue(position)

        for i in range(0, N):
            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        velocity = w * velocity + (
                    c1 * np.random.rand(N, D) * (Pb - position) + c2 * np.random.rand(N, D) * (Gb[0] - position))

        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    #print(bestFitness)
    #print(Gb)
    return [fitData, Gb]


def SFPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c1 = 1.494
    c2 = 1.494
    w = 0.729

    BA = nx.generators.random_graphs.barabasi_albert_graph2(N, 4, 4)
    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    lastFitness = np.zeros(N) + 999999999999999
    bestFitness = 999999999999999
    neighboorFitness = np.zeros(N) + 999999999999999
    neighboor = []
    Pb = np.zeros([N, D])
    Lb = np.zeros([N, D])
    fitData = []

    i = 0
    for node in nx.nodes(BA):
        neighboor.append([])
        for j in nx.all_neighbors(BA, node):
            neighboor[i].append(j)
        i += 1

    t = 0
    while t < T:

        if np.amin(position) < benchmark.low or np.amax(position) > benchmark.high:
            for i in range(0, N):
                for j in range(0, D):
                    while position[i][j] < benchmark.low:
                        position[i][j] = benchmark.low
                    while position[i][j] > benchmark.high:
                        position[i][j] = benchmark.high

        fitness = benchmark.getValue(position)

        for i in range(0, N):
            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness

        for index in range(0, N):
            for j in neighboor[index]:
                if fitness[j] < neighboorFitness[index]:
                    neighboorFitness[index] = fitness[j]
                    Lb[index] = position[j]
        velocity = w * velocity + (
                    c1 * np.random.rand(N, D) * (Pb - position) + c2 * np.random.rand(N, D) * (Lb - position))
        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    'print(bestFitness)'
    return [fitData, Gb]
