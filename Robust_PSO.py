import numpy as np
import networkx as nx


def checkBoundary(position, low, high):
    if np.amin(position) < low or np.amax(position) > high:
        for i in range(0, position.shape[0]):
            for j in range(0, position.shape[1]):
                while position[i][j] < low:
                    position[i][j] = low
                while position[i][j] > high:
                    position[i][j] = high


def GPSO(benchmark, T, m, e):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298

    biasPos = []
    biasFit = []

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

        for i in range(0, N):
            biasPos.append(np.zeros([m, D]))
            biasFit.append(np.zeros(m))
            for j in range(0, m):
                biasPos[i][j] = position[i] + (np.random.rand(D) * 2 * e - e)

            checkBoundary(biasPos[i], benchmark.low, benchmark.high)
            biasFit[i] = benchmark.getValue(biasPos[i])

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
            # use max fitness
            temp = max(biasFit[i])
            fitness[i] = max(temp, fitness[i])

            # use average fitness
            # for one in range(0, m):
            #     fitness[i] += biasFit[i][one]
            # fitness[i] /= (m+1)

            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        velocity = w * (velocity + c1 * np.random.rand(N, D) * (Pb - position)
                        + c2 * np.random.rand(N, D) * (Gb[0] - position))

        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    # print(bestFitness)
    # print(Gb)
    return [fitData, Gb]


def LPSO(benchmark, T, m, e):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298

    biasPos = []
    biasFit = []

    graph = nx.generators.random_graphs.watts_strogatz_graph(N, 2, 0)
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
    for node in nx.nodes(graph):
        neighboor.append([])
        for j in nx.all_neighbors(graph, node):
            neighboor[i].append(j)
        i += 1

    t = 0
    while t < T:

        for i in range(0, N):
            biasPos.append(np.zeros([m, D]))
            biasFit.append(np.zeros(m))
            for j in range(0, m):
                biasPos[i][j] = position[i] + (np.random.rand(D) * 2 * e - e)

            checkBoundary(biasPos[i], benchmark.low, benchmark.high)
            biasFit[i] = benchmark.getValue(biasPos[i])

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
            # use max fitness
            temp = max(biasFit[i])
            fitness[i] = max(temp, fitness[i])

            # use average fitness
            # for one in range(0, m):
            #     fitness[i] += biasFit[i][one]
            # fitness[i] /= (m + 1)

            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        for index in range(0, N):
            for j in neighboor[index]:
                if fitness[j] < neighboorFitness[index]:
                    neighboorFitness[index] = fitness[j]
                    Lb[index] = position[j]
        velocity = w * (velocity + c1 * np.random.rand(N, D) * (Pb - position)
                        + c2 * np.random.rand(N, D) * (Lb - position))
        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    # print(bestFitness)
    # print(Gb)
    return [fitData, Gb]


def SFPSO(benchmark, T, m, e):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298

    biasPos = []
    biasFit = []

    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 4, 2)
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
    for node in nx.nodes(graph):
        neighboor.append([])
        for j in nx.all_neighbors(graph, node):
            neighboor[i].append(j)
        i += 1

    t = 0
    while t < T:

        for i in range(0, N):
            biasPos.append(np.zeros([m, D]))
            biasFit.append(np.zeros(m))
            for j in range(0, m):
                biasPos[i][j] = position[i] + (np.random.rand(D) * 2 * e - e)

            checkBoundary(biasPos[i], benchmark.low, benchmark.high)
            biasFit[i] = benchmark.getValue(biasPos[i])

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
            # use max fitness
            temp = max(biasFit[i])
            fitness[i] = max(temp, fitness[i])

            # use average fitness
            # for one in range(0, m):
            #     fitness[i] += biasFit[i][one]
            # fitness[i] /= (m + 1)

            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        for index in range(0, N):
            for j in neighboor[index]:
                if fitness[j] < neighboorFitness[index]:
                    neighboorFitness[index] = fitness[j]
                    Lb[index] = position[j]
        velocity = w * (velocity + c1 * np.random.rand(N, D) * (Pb - position)
                        + c2 * np.random.rand(N, D) * (Lb - position))
        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    # print(bestFitness)
    # print(Gb)
    return [fitData, Gb]


def LFIPSO(benchmark, T, m, e):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    fi = c1 + c2

    biasPos = []
    biasFit = []

    graph = nx.generators.random_graphs.watts_strogatz_graph(N, 2, 0)
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
    for node in nx.nodes(graph):
        neighboor.append([])
        for j in nx.all_neighbors(graph, node):
            neighboor[i].append(j)
        i += 1

    t = 0
    while t < T:

        for i in range(0, N):
            biasPos.append(np.zeros([m, D]))
            biasFit.append(np.zeros(m))
            for j in range(0, m):
                biasPos[i][j] = position[i] + (np.random.rand(D) * 2 * e - e)

            checkBoundary(biasPos[i], benchmark.low, benchmark.high)
            biasFit[i] = benchmark.getValue(biasPos[i])

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
            # use max fitness
            temp = max(biasFit[i])
            fitness[i] = max(temp, fitness[i])

            # use average fitness
            # for one in range(0, m):
            #     fitness[i] += biasFit[i][one]
            # fitness[i] /= (m + 1)

            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        for index in range(0, N):
            for j in neighboor[index]:
                Lb[index] = 0
                Lb[index] += position[j] * np.random.rand(D) * fi
            Lb[index] /= neighboor[index].__len__()

        velocity = w * (velocity + Lb)
        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    # print(bestFitness)
    # print(Gb)
    return [fitData, Gb]


def SFIPSO(benchmark, T, m, e):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    fi = c1 + c2

    biasPos = []
    biasFit = []

    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 4, 2)
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
    for node in nx.nodes(graph):
        neighboor.append([])
        for j in nx.all_neighbors(graph, node):
            neighboor[i].append(j)
        i += 1

    t = 0
    while t < T:

        for i in range(0, N):
            biasPos.append(np.zeros([m, D]))
            biasFit.append(np.zeros(m))
            for j in range(0, m):
                biasPos[i][j] = position[i] + (np.random.rand(D) * 2 * e - e)

            checkBoundary(biasPos[i], benchmark.low, benchmark.high)
            biasFit[i] = benchmark.getValue(biasPos[i])

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
            # use max fitness
            temp = max(biasFit[i])
            fitness[i] = max(temp, fitness[i])

            # use average fitness
            # for one in range(0, m):
            #     fitness[i] += biasFit[i][one]
            # fitness[i] /= (m + 1)

            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        for index in range(0, N):
            for j in neighboor[index]:
                Lb[index] = 0
                Lb[index] += position[j] * np.random.rand(D) * fi
            Lb[index] /= neighboor[index].__len__()

        velocity = w * (velocity + Lb)
        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    # print(bestFitness)
    # print(Gb)
    return [fitData, Gb]


def SIPSO(benchmark, T, m, e):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    fi = c1 + c2
    kc = 3

    biasPos = []
    biasFit = []

    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 4, 2)
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
    for node in nx.nodes(graph):
        neighboor.append([])
        for j in nx.all_neighbors(graph, node):
            neighboor[i].append(j)
        i += 1

    t = 0
    while t < T:

        for i in range(0, N):
            biasPos.append(np.zeros([m, D]))
            biasFit.append(np.zeros(m))
            for j in range(0, m):
                biasPos[i][j] = position[i] + (np.random.rand(D) * 2 * e - e)

            checkBoundary(biasPos[i], benchmark.low, benchmark.high)
            biasFit[i] = benchmark.getValue(biasPos[i])

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
            # use max fitness
            temp = max(biasFit[i])
            fitness[i] = max(temp, fitness[i])

            # use average fitness
            # for one in range(0, m):
            #     fitness[i] += biasFit[i][one]
            # fitness[i] /= (m + 1)

            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        for index in range(0, N):
            if neighboor[index].__len__() > kc:
                for j in neighboor[index]:
                    Lb[index] = 0
                    Lb[index] += position[j] * np.random.rand(D) * fi
                Lb[index] /= neighboor[index].__len__()
                velocity[index] = w * (velocity[index] + Lb[index])
            else:
                velocity[index] = w * (velocity[index] + c1 * np.random.rand(D) * (Pb[index] - position[index])
                                       + c2 * np.random.rand(D) * (Gb[0] - position[index]))

        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    # print(bestFitness)
    # print(Gb)
    return [fitData, Gb]


def SWPSO(benchmark, T, m, e):
    N = benchmark.N
    D = benchmark.D
    c1 = 1.494
    c2 = 1.494
    w = 0.729

    biasPos = []
    biasFit = []

    graph = nx.generators.random_graphs.watts_strogatz_graph(N, 4, 0.3)
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
    for node in nx.nodes(graph):
        neighboor.append([])
        for j in nx.all_neighbors(graph, node):
            neighboor[i].append(j)
        i += 1

    t = 0
    while t < T:

        for i in range(0, N):
            biasPos.append(np.zeros([m, D]))
            biasFit.append(np.zeros(m))
            for j in range(0, m):
                biasPos[i][j] = position[i] + (np.random.rand(D) * 2 * e - e)

            checkBoundary(biasPos[i], benchmark.low, benchmark.high)
            biasFit[i] = benchmark.getValue(biasPos[i])

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
            # use max fitness
            temp = max(biasFit[i])
            fitness[i] = max(temp, fitness[i])

            # use average fitness
            # for one in range(0, m):
            #     fitness[i] += biasFit[i][one]
            # fitness[i] /= (m + 1)

            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

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
    # print(bestFitness)
    # print(Gb)
    return [fitData, Gb]
