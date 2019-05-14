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

def GPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298

    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    fitness = np.zeros(N)
    lastFitness = np.zeros(N) + float('inf')
    bestFitness = float('inf')
    Pb = np.zeros([N, D])
    Gb = np.zeros(D)
    fitData = []

    t = 0
    while t < T:
        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
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


def LPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298

    graph = nx.generators.random_graphs.watts_strogatz_graph(N, 2, 0)
    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    lastFitness = np.zeros(N) + float('inf')
    bestFitness = float('inf')
    neighboorFitness = np.zeros(N) + float('inf')
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

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):

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


def SFPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298

    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 4, 2)
    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    lastFitness = np.zeros(N) + float('inf')
    bestFitness = float('inf')
    neighboorFitness = np.zeros(N) + float('inf')
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

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
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


def LFIPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    fi = c1 + c2

    graph = nx.generators.random_graphs.watts_strogatz_graph(N, 2, 0)
    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    lastFitness = np.zeros(N) + float('inf')
    bestFitness = float('inf')
    neighboorFitness = np.zeros(N) + float('inf')
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
        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
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


def SFIPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    fi = c1 + c2

    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 4, 2)
    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    lastFitness = np.zeros(N) + float('inf')
    bestFitness = float('inf')
    neighboorFitness = np.zeros(N) + float('inf')
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
        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
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


def SIPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    fi = c1 + c2
    kc = 3

    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 4, 2)
    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    lastFitness = np.zeros(N) + float('inf')
    bestFitness = float('inf')
    neighboorFitness = np.zeros(N) + float('inf')
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

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N):
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


def CLPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    c = 1.49445
    w0 = 0.9
    w1 = 0.4
    m = 7
    pc = np.zeros(N)
    position = np.random.rand(N, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N, D])
    fitness = np.zeros(N)
    lastFitness = np.zeros(N) + float('inf')
    bestFitness = float('inf')
    Pb = np.zeros([N, D])
    Pbs = np.zeros([N, D])
    Gb = np.zeros(D)
    fitData = []
    flags = np.zeros(N)
    Vmax = (benchmark.high - benchmark.low)/5
    for i in range(0, N):
        pc[i] = 0.05 + 0.45*((np.e**(10*i/(N - 1))-1)/(np.e**10-1))

    t = 0
    while t < T:
        w = w0-((w0-w1)*t/T)

        fitness = benchmark.getValue(position)

        for i in range(0, N):
            if np.any(position[i] > benchmark.high) or np.any(position[i] < benchmark.low):
                flags[i] += 1
                continue

            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]
                flags[i] = 0
            else:
                flags[i] += 1

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        for i in range(0, N):
            if flags[i] >= m:
                flag = True
                for j in range(0, D):
                    rand = np.random.rand()
                    if rand < pc[i]:
                        flag = False
                        rand1 = np.random.randint(0, N)
                        rand2 = np.random.randint(0, N)
                        while rand1 == rand2 or rand1 == i or rand2 == i:
                            rand1 = np.random.randint(0, N)
                            rand2 = np.random.randint(0, N)

                        if lastFitness[rand1] < lastFitness[rand2]:
                            Pbs[i][j] = Pb[rand1][j]
                        else:
                            Pbs[i][j] = Pb[rand2][j]
                    else:
                        Pbs[i][j] = Pb[i][j]
                if flag:
                    rand3 = np.random.randint(0, N)
                    while rand3 == i:
                        rand3 = np.random.randint(0, N)
                    rand4 = np.random.randint(0, D)
                    Pbs[i][rand4] = Pb[rand3][rand4]

        velocity = w * velocity + c * np.random.rand(N, D) * (Pbs - position)
        velocity[velocity > Vmax] = Vmax
        velocity[velocity < -Vmax] = -Vmax
        position += velocity

        t += 1
        fitData.append(bestFitness)
        'print(bestFitness)'
    print(bestFitness)
    # print(Gb)
    return [fitData, Gb]


def MPPSO(benchmark, T, N=80):
    N0 = benchmark.N
    D = benchmark.D
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    tGap = 4

    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 5, 2)
    position = np.random.rand(N0, D) * (benchmark.high - benchmark.low) + benchmark.low
    velocity = np.zeros([N0, D])
    indexInSwarmNetwork = np.zeros(N, dtype=int) - 1
    counter = np.zeros(N0)
    lastFitness = np.zeros(N0) + float('inf')
    bestFitness = float('inf')
    neighboorFitness = np.zeros(N0) + float('inf')
    neighboor = []
    Pb = np.zeros([N0, D])
    Lb = np.zeros([N0, D])
    fitData = []

    i = 0
    for node in nx.nodes(graph):
        neighboor.append([])
        for j in nx.all_neighbors(graph, node):
            neighboor[i].append(j)
        i += 1

    # assign swarms in baseNetwork
    randList = list(range(0, N))
    indexInBaseNetwork = np.random.choice(randList, N0, replace=False)
    for i in range(0, N0):
        indexInSwarmNetwork[indexInBaseNetwork[i]] = i


    t = 0
    while t < T:

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness = benchmark.getValue(position)

        for i in range(0, N0):
            if fitness[i] < lastFitness[i]:
                lastFitness[i] = fitness[i]
                Pb[i] = position[i]
                counter[i] = 0
            else:
                counter[i] += 1

        currentBestFitness = min(lastFitness)
        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            Gb = position[np.where(fitness == np.min(fitness))[0]]

        for i in range(0, N0):
            for j in neighboor[indexInBaseNetwork[i]]:
                if indexInSwarmNetwork[j] != -1:
                    if fitness[indexInSwarmNetwork[j]] < neighboorFitness[i]:
                        neighboorFitness[i] = fitness[indexInSwarmNetwork[j]]
                        Lb[i] = position[indexInSwarmNetwork[j]]

        velocity = w * (velocity + c1 * np.random.rand(N0, D) * (Pb - position)
                        + c2 * np.random.rand(N0, D) * (Lb - position))
        position += velocity

        for i in np.random.choice(list(range(0, N0)), N0, replace=False):
            if counter[i] > tGap:
                chooseList = []
                # only change indexInBaseNetwork and indexInSwarmNetwork
                for j in neighboor[indexInBaseNetwork[i]]:
                    if indexInSwarmNetwork[j] == -1:
                        chooseList.append(j)
                if len(chooseList) != 0:
                    indexInSwarmNetwork[indexInBaseNetwork[i]] = -1
                    indexInBaseNetwork[i] = np.random.choice(chooseList)
                    indexInSwarmNetwork[indexInBaseNetwork[i]] = i
                    counter[i] = 0
        t += 1
        fitData.append(bestFitness)
        print(bestFitness)
    # print(bestFitness)
    # print(Gb)
    return [fitData, Gb]

