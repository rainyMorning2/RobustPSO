import numpy as np
import Repository
import networkx as nx

def checkBoundary(position, velocity, low, high):
    for i in range(0, position.shape[0]):
        for j in range(0, position.shape[1]):
            if position[i][j] < low[j]:
                position[i][j] = low[j]
                velocity[i] *= -1
            if position[i][j] > high[j]:
                position[i][j] = high[j]
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
    posBound = benchmark.high - benchmark.low
    position = np.random.rand(N, D) * posBound + benchmark.low
    velocity = np.zeros([N, D])
    repo = Repository.Repository(repsize, divisions)
    pb = position
    storedFit, storedCons = benchmark.getValue(pb)

    t = 0
    while t < T:

        # check boundary
        checkBoundary(position, velocity, benchmark.low, benchmark.high)

        # mutation
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

        # evaluate particles
        currentFit, currentCons = benchmark.getValue(position)

        # update repo
        for j in range(0, K):
            if np.any(currentCons == j):
                for i in np.where(currentCons == j)[0]:
                    repo.insert(currentFit[i], position[i], j)
                break
        repo.updateGrid()

        # update pb
        for i in range(0, N):
            if currentCons[i] < storedCons[i]:
                storedCons[i] = currentCons[i]
                storedFit[i] = currentFit[i]
                pb[i] = position[i]
            elif currentCons[i] == storedCons[i]:
                if isDominated(currentFit[i], storedFit[i]):
                    storedCons[i] = currentCons[i]
                    storedFit[i] = currentFit[i]
                    pb[i] = position[i]
                elif isDominated(storedFit[i], currentFit[i]):
                    pass
                else:
                    if np.random.rand() > 0.5:
                        storedCons[i] = currentCons[i]
                        storedFit[i] = currentFit[i]
                        pb[i] = position[i]

        # update vel
        velocity += w * velocity + np.random.rand(N, D) * (pb - position) + np.random.rand(N, D) * (repo.get() - position)
        velocity = np.maximum(velocity, -posBound / 3)
        velocity = np.minimum(velocity, posBound / 3)
        # velocity[velocity > 4] = 4
        # velocity[velocity < -4] = -4
        # update position
        position += velocity
        t += 1
        print(t)
    return repo.getAll()


def MOSFPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    K = benchmark.K
    C = benchmark.C
    w = 0.4
    mutrate = 0.5

    repsize = np.zeros(N)
    divisions = np.zeros(N)
    posBound = benchmark.high - benchmark.low
    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 5, 3)

    position = np.random.rand(N, D) * posBound + benchmark.low
    velocity = np.zeros([N, D])
    neighbor = []

    pb = position
    lbest = np.zeros([N, D])
    storedFit, storedCons = benchmark.getValue(pb)

    i = 0
    for node in nx.nodes(graph):
        neighbor.append([])
        for j in nx.all_neighbors(graph, node):
            neighbor[i].append(j)
            # repsize[i] = 100
            # divisions[i] = 30
        if neighbor[i].__len__() < 7:
            repsize[i] = 10
            divisions[i] = 10
        else:
            repsize[i] = neighbor.__len__()
            divisions[i] = 10
        i += 1

    repos = []
    for i in range(0, N):
        repos.append(Repository.Repository(repsize[i], divisions[i]))

    t = 0
    while t < T:

        # check boundary
        checkBoundary(position, velocity, benchmark.low, benchmark.high)

        # mutation
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

        # evaluate particles
        currentFit, currentCons = benchmark.getValue(position)

        # update repo !!
        for i in range(0, N):
            for j in range(0, C+1):
                if np.any(currentCons[neighbor[i]] == j):
                    for k in np.where(currentCons[neighbor[i]] == j)[0]:
                        repos[i].insert(currentFit[k], position[k], j)
                    break
            repos[i].updateGrid()
            lbest[i] = repos[i].get()

        # update pb
        for i in range(0, N):
            if currentCons[i] < storedCons[i]:
                storedCons[i] = currentCons[i]
                storedFit[i] = currentFit[i]
                pb[i] = position[i]
            elif currentCons[i] == storedCons[i]:
                if isDominated(currentFit[i], storedFit[i]):
                    storedCons[i] = currentCons[i]
                    storedFit[i] = currentFit[i]
                    pb[i] = position[i]
                elif isDominated(storedFit[i], currentFit[i]):
                    pass
                else:
                    if np.random.rand() > 0.5:
                        storedCons[i] = currentCons[i]
                        storedFit[i] = currentFit[i]
                        pb[i] = position[i]

        # update vel
        velocity += w * velocity + np.random.rand(N, D) * (pb - position) + np.random.rand(N, D) * (lbest - position)
        velocity = np.maximum(velocity, -posBound / 5)
        velocity = np.minimum(velocity, posBound / 5)
        # update position
        position += velocity
        t += 1
        # print(t)

    repo = Repository.Repository(200)
    for re in repos:
        for item in re.getAll():
            repo.insert(item, None, re.cons)
    return repo.getAll()


def MOCLPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    K = benchmark.K
    w0 = 0.9
    w1 = 0.4
    m = 7
    c = 1.49445
    pc = np.zeros(N)
    mutrate = 0.5

    repsize = 100
    divisions = 30
    posBound = benchmark.high - benchmark.low
    position = np.random.rand(N, D) * posBound + benchmark.low
    velocity = np.zeros([N, D])
    flags = np.zeros(N)
    repo = Repository.Repository(repsize, divisions)
    pb = position
    Pbs = np.zeros([N, D])
    storedFit, storedCons = benchmark.getValue(pb)

    for i in range(0, N):
        pc[i] = 0.05 + 0.45*((np.e**(10*i/(N - 1))-1)/(np.e**10-1))

    t = 0
    while t < T:
        w = w0 - ((w0 - w1) * t / T)
        # evaluate particles
        currentFit, currentCons = benchmark.getValue(position)

        # update pb
        for i in range(0, N):
            if np.any(position[i] > benchmark.high) or np.any(position[i] < benchmark.low):
                flags[i] += 1
                continue
            if currentCons[i] < storedCons[i]:
                storedCons[i] = currentCons[i]
                storedFit[i] = currentFit[i]
                pb[i] = position[i]
                flags[i] = 0
            elif currentCons[i] == storedCons[i]:
                if isDominated(currentFit[i], storedFit[i]):
                    storedCons[i] = currentCons[i]
                    storedFit[i] = currentFit[i]
                    pb[i] = position[i]
                    flags[i] = 0
                elif isDominated(storedFit[i], currentFit[i]):
                    flags[i] += 1
                else:
                    if np.random.rand() > 0.5:
                        storedCons[i] = currentCons[i]
                        storedFit[i] = currentFit[i]
                        pb[i] = position[i]
                        flags[i] = 0
                    else:
                        flags[i] += 1

        # mutation
        # if (1 - 1.0 * t / T) ** (5.0 / mutrate) != 0:
        #     for i in range(0, N):
        #         dim = np.random.randint(0, D)
        #         mutrange = (benchmark.high[dim] - benchmark.low[dim]) * (1 - 1.0 * t / T) ** (5.0 / mutrate)
        #         hb = position[i][dim] + mutrange
        #         lb = position[i][dim] - mutrange
        #
        #         if hb > benchmark.high[dim]:
        #             hb = benchmark.high[dim]
        #         if lb < benchmark.low[dim]:
        #             lb = benchmark.low[dim]
        #         position[i][dim] = np.random.rand() * (hb - lb) + lb

        # update repo
        for j in range(0, K):
            if np.any(currentCons == j):
                for i in np.where(currentCons == j)[0]:
                    if np.any(np.isnan(currentFit[i])):
                        continue
                    else:
                        repo.insert(currentFit[i], position[i], j)
                break
        repo.updateGrid()

        # get Pbs
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

                        if storedCons[rand1] < storedCons[rand2]:
                            Pbs[i][j] = pb[rand1][j]
                        elif storedCons[rand2] < storedCons[rand1]:
                            Pbs[i][j] = pb[rand2][j]
                        else:
                            if isDominated(storedFit[rand1], storedFit[rand2]):
                                Pbs[i][j] = pb[rand1][j]
                            elif isDominated(storedFit[rand2], storedFit[rand1]):
                                Pbs[i][j] = pb[rand2][j]
                            else:
                                if np.random.rand() < 0.5:
                                    Pbs[i][j] = pb[rand1][j]
                                else:
                                    Pbs[i][j] = pb[rand2][j]
                    else:
                        Pbs[i][j] = pb[i][j]
                if flag:
                    rand3 = np.random.randint(0, N)
                    while rand3 == i:
                        rand3 = np.random.randint(0, N)
                    rand4 = np.random.randint(0, D)
                    Pbs[i][rand4] = pb[rand3][rand4]

        # update vel
        velocity = w * velocity + c * np.random.rand(N, D) * (Pbs - position)
        velocity = np.maximum(velocity, -posBound / 3)
        velocity = np.minimum(velocity, posBound / 3)
        # velocity[velocity > 4] = 4
        # velocity[velocity < -4] = -4

        # update position
        position += velocity
        t += 1
        print(t)
    return repo.getAll()


def MOSIPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    K = benchmark.K
    C = benchmark.C
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    fi = c1 + c2
    kc = 5
    mutrate = 0.5

    repsize = np.zeros(N)
    divisions = np.zeros(N)
    posBound = benchmark.high - benchmark.low
    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 4, 2)

    position = np.random.rand(N, D) * posBound + benchmark.low
    velocity = np.zeros([N, D])
    neighbor = []
    Lb = np.zeros([N, D])
    pb = position
    lbest = np.zeros([N, D])
    storedFit, storedCons = benchmark.getValue(pb)

    i = 0
    for node in nx.nodes(graph):
        neighbor.append([])
        for j in nx.all_neighbors(graph, node):
            neighbor[i].append(j)
        if neighbor[i].__len__() < 5:
            repsize[i] = 5
            divisions[i] = 2
        else:
            repsize[i] = neighbor.__len__()
            divisions[i] = 30
        i += 1

    repos = []
    for i in range(0, N):
        repos.append(Repository.Repository(repsize[i], divisions[i]))

    t = 0
    while t < T:

        # check boundary
        checkBoundary(position, velocity, benchmark.low, benchmark.high)

        # mutation
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

        # evaluate particles
        currentFit, currentCons = benchmark.getValue(position)

        # update repo !!
        for i in range(0, N):
            for j in range(0, C+1):
                if np.any(currentCons[neighbor[i]] == j):
                    for k in np.where(currentCons[neighbor[i]] == j)[0]:
                        repos[i].insert(currentFit[k], position[k], j)
                    break
            repos[i].updateGrid()
            lbest[i] = repos[i].get()

        # update pb
        for i in range(0, N):
            if currentCons[i] < storedCons[i]:
                storedCons[i] = currentCons[i]
                storedFit[i] = currentFit[i]
                pb[i] = position[i]
            elif currentCons[i] == storedCons[i]:
                if isDominated(currentFit[i], storedFit[i]):
                    storedCons[i] = currentCons[i]
                    storedFit[i] = currentFit[i]
                    pb[i] = position[i]
                elif isDominated(storedFit[i], currentFit[i]):
                    pass
                else:
                    if np.random.rand() > 0.5:
                        storedCons[i] = currentCons[i]
                        storedFit[i] = currentFit[i]
                        pb[i] = position[i]

        # update vel
        for index in range(0, N):
            if neighbor[index].__len__() > kc:
                for j in neighbor[index]:
                    Lb[index] = 0
                    Lb[index] += (pb[j]-position[index]) * np.random.rand(D) * fi
                Lb[index] /= neighbor[index].__len__()
                velocity[index] = w * (velocity[index] + Lb[index])
            else:
                velocity[index] = w * (velocity[index] + c1 * np.random.rand(D) * (pb[index] - position[index])
                                       + c2 * np.random.rand(D) * (lbest[index] - position[index]))

        # velocity = np.maximum(velocity, -posBound / 3)
        # velocity = np.minimum(velocity, posBound / 3)

        # update position
        position += velocity
        t += 1
        print(t)

    repo = Repository.Repository(200)
    for re in repos:
        for item in re.getAll():
            repo.insert(item, None, re.cons)
    return repo.getAll()


def MOMPPSO(benchmark, T, N=80):
    N0 = benchmark.N
    D = benchmark.D
    K = benchmark.K
    C = benchmark.C
    c1 = 2.05
    c2 = 2.05
    w = 0.7298
    tGap = 4
    mutrate = 0.5

    repsize = np.zeros(N0)
    divisions = np.zeros(N0)
    posBound = benchmark.high - benchmark.low
    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 5, 2)

    position = np.random.rand(N0, D) * posBound + benchmark.low
    velocity = np.zeros([N0, D])
    neighbor = []
    indexInSwarmNetwork = np.zeros(N, dtype=int) - 1
    counter = np.zeros(N0)
    pb = position
    lbest = np.zeros([N0, D])
    storedFit, storedCons = benchmark.getValue(pb)

    i = 0
    for node in nx.nodes(graph):
        neighbor.append([])
        for j in nx.all_neighbors(graph, node):
            neighbor[i].append(j)
        i += 1

    # assign swarms in baseNetwork
    randList = list(range(0, N))
    indexInBaseNetwork = np.random.choice(randList, N0, replace=False)
    for i in range(0, N0):
        indexInSwarmNetwork[indexInBaseNetwork[i]] = i

    for i in range(0, N0):
        if neighbor[indexInBaseNetwork[i]].__len__() < 5:
            repsize[i] = 10
            divisions[i] = 10
        else:
            repsize[i] = 20
            divisions[i] = 10

    repos = []
    for i in range(0, N0):
        repos.append(Repository.Repository(repsize[i], divisions[i]))

    t = 0
    while t < T:

        # check boundary
        checkBoundary(position, velocity, benchmark.low, benchmark.high)

        # mutation
        if (1 - 1.0 * t / T) ** (5.0 / mutrate) != 0:
            for i in range(0, N0):
                dim = np.random.randint(0, D)
                mutrange = (benchmark.high[dim] - benchmark.low[dim]) * (1 - 1.0 * t / T) ** (5.0 / mutrate)
                hb = position[i][dim] + mutrange
                lb = position[i][dim] - mutrange

                if hb > benchmark.high[dim]:
                    hb = benchmark.high[dim]
                if lb < benchmark.low[dim]:
                    lb = benchmark.low[dim]
                position[i][dim] = np.random.rand() * (hb - lb) + lb

        # evaluate particles
        currentFit, currentCons = benchmark.getValue(position)

        # update repo !!
        for i in range(0, N0):
            for j in range(0, C+1):
                if np.any(currentCons[indexInSwarmNetwork[neighbor[indexInBaseNetwork[i]]]] == j):
                    for k in indexInSwarmNetwork[neighbor[indexInBaseNetwork[i]]]:
                        if k != -1 and currentCons[k] == j:
                            repos[i].insert(currentFit[k], position[k], j)
                    break
            if repos[i].size() != 0:
                repos[i].updateGrid()
                lbest[i] = repos[i].get()

        # update pb
        for i in range(0, N0):
            if currentCons[i] < storedCons[i]:
                storedCons[i] = currentCons[i]
                storedFit[i] = currentFit[i]
                pb[i] = position[i]
                counter[i] = 0
            elif currentCons[i] == storedCons[i]:
                if isDominated(currentFit[i], storedFit[i]):
                    storedCons[i] = currentCons[i]
                    storedFit[i] = currentFit[i]
                    pb[i] = position[i]
                    counter[i] = 0
                elif isDominated(storedFit[i], currentFit[i]):
                    counter[i] += 1
                else:
                    if np.random.rand() > 0.5:
                        storedCons[i] = currentCons[i]
                        storedFit[i] = currentFit[i]
                        pb[i] = position[i]
                        counter[i] = 0

        # update vel
        velocity += w * (velocity + c1 * np.random.rand(N0, D) * (pb - position)
                        + c2 * np.random.rand(N0, D) * (lbest - position))
        # velocity = np.maximum(velocity, -posBound / 3)
        # velocity = np.minimum(velocity, posBound / 3)
        # update position
        position += velocity

        for i in np.random.choice(list(range(0, N0)), N0, replace=False):
            if counter[i] > tGap:
                chooseList = []
                # only change indexInBaseNetwork and indexInSwarmNetwork
                for j in neighbor[indexInBaseNetwork[i]]:
                    if indexInSwarmNetwork[j] == -1:
                        chooseList.append(j)
                if len(chooseList) != 0:
                    indexInSwarmNetwork[indexInBaseNetwork[i]] = -1
                    indexInBaseNetwork[i] = np.random.choice(chooseList)
                    indexInSwarmNetwork[indexInBaseNetwork[i]] = i
                    counter[i] = 0

        t += 1
        # print(t)

    repo = Repository.Repository(200)
    for re in repos:
        for item in re.getAll():
            repo.insert(item, None, re.cons)
    return repo.getAll()


def MOHCLPSO(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    K = benchmark.K
    w0 = 0.9
    w1 = 0.4
    m = 7
    c = 1.49445
    pc = np.zeros(N)
    mutrate = 0.5

    graph = nx.generators.random_graphs.barabasi_albert_graph2(N, 5, 4)

    repsize = 100
    divisions = 30
    neighbor = []
    posBound = benchmark.high - benchmark.low
    position = np.random.rand(N, D) * posBound + benchmark.low
    velocity = np.zeros([N, D])
    flags = np.zeros(N)
    repo = Repository.Repository(repsize, divisions)
    pb = position
    Pbs = np.zeros([N, D])
    storedFit, storedCons = benchmark.getValue(pb)

    i = 0
    for node in nx.nodes(graph):
        neighbor.append([])
        for j in nx.all_neighbors(graph, node):
            neighbor[i].append(j)
        i += 1

    for i in range(0, N):
        pc[i] = 1 / (1 + np.e**(8/neighbor[i].__len__()))

    t = 0
    while t < T:
        w = w0 - ((w0 - w1) * t / T)
        # evaluate particles
        currentFit, currentCons = benchmark.getValue(position)

        # update pb
        for i in range(0, N):
            if np.any(position[i] > benchmark.high) or np.any(position[i] < benchmark.low):
                flags[i] += 1
                continue
            if currentCons[i] < storedCons[i]:
                storedCons[i] = currentCons[i]
                storedFit[i] = currentFit[i]
                pb[i] = position[i]
                flags[i] = 0
            elif currentCons[i] == storedCons[i]:
                if isDominated(currentFit[i], storedFit[i]):
                    storedCons[i] = currentCons[i]
                    storedFit[i] = currentFit[i]
                    pb[i] = position[i]
                    flags[i] = 0
                elif isDominated(storedFit[i], currentFit[i]):
                    flags[i] += 1
                else:
                    if np.random.rand() > 0.5:
                        storedCons[i] = currentCons[i]
                        storedFit[i] = currentFit[i]
                        pb[i] = position[i]
                        flags[i] = 0
                    else:
                        flags[i] += 1

        # mutation
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

        # update repo
        for j in range(0, K):
            if np.any(currentCons == j):
                for i in np.where(currentCons == j)[0]:
                    if np.any(np.isnan(currentFit[i])):
                        continue
                    else:
                        repo.insert(currentFit[i], position[i], j)
                break
        repo.updateGrid()

        # get Pbs
        for i in range(0, N):
            if flags[i] >= m:
                if neighbor[i].__len__() < 8:
                    # two particles competition
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

                            if storedCons[rand1] < storedCons[rand2]:
                                Pbs[i][j] = pb[rand1][j]
                            elif storedCons[rand2] < storedCons[rand1]:
                                Pbs[i][j] = pb[rand2][j]
                            else:
                                if isDominated(storedFit[rand1], storedFit[rand2]):
                                    Pbs[i][j] = pb[rand1][j]
                                elif isDominated(storedFit[rand2], storedFit[rand1]):
                                    Pbs[i][j] = pb[rand2][j]
                                else:
                                    if np.random.rand() < 0.5:
                                        Pbs[i][j] = pb[rand1][j]
                                    else:
                                        Pbs[i][j] = pb[rand2][j]
                        else:
                            Pbs[i][j] = pb[i][j]
                    if flag:
                        rand3 = np.random.randint(0, N)
                        while rand3 == i:
                            rand3 = np.random.randint(0, N)
                        rand4 = np.random.randint(0, D)
                        Pbs[i][rand4] = pb[rand3][rand4]
                else:
                    # four particles competition
                    flag = True
                    for j in range(0, D):
                        rand = np.random.rand()
                        if rand < pc[i]:
                            flag = False
                            rand0 = np.random.choice(list(range(0, N)), 4, False)
                            while np.any(rand0 == i):
                                rand0 = np.random.choice(list(range(0, N)), 4, False)
                            repoTemp = Repository.Repository(4)
                            for k in rand0:
                                repoTemp.insert(storedCons[k], pb[k], storedCons[k])
                            repoTemp.updateGrid()
                            Pbs[i][j] = repoTemp.get()[j]
                        else:
                            Pbs[i][j] = pb[i][j]
                    if flag:
                        rand5 = np.random.randint(0, N)
                        while rand5 == i:
                            rand5 = np.random.randint(0, N)
                        rand6 = np.random.randint(0, D)
                        Pbs[i][rand6] = pb[rand5][rand6]

        # update vel
        velocity = w * velocity + c * np.random.rand(N, D) * (Pbs - position)
        velocity = np.maximum(velocity, -posBound / 5)
        velocity = np.minimum(velocity, posBound / 5)

        # update position
        position += velocity
        t += 1
        print(t)
    return repo.getAll()
