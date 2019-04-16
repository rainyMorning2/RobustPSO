import numpy as np
import copy

DEFAULTSIZE = 4


def isCrowdedDominated(index1, index2, rank, distance):
    if rank[index1] < rank[index2] or (rank[index1] == rank[index2] and distance[index1] > distance[index2]):
        return True
    else:
        return False


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


# distance needs to be init as np.zeros(N)
def crowdingDistanceAssignment(subfitness, distance):
    for i in range(0, len(subfitness[0][1])):
        subfitness.sort(key=lambda x: x[1][i])
        distance[subfitness[0][0]] = float("inf")
        distance[subfitness[-1][0]] = float("inf")

        Vmin = subfitness[0][1][i]
        Vmax = subfitness[-1][1][i]

        for j in range(1, len(subfitness) - 1):
            if Vmin == Vmax:
                distance[subfitness[j][0]] = float("inf")
            else:
                distance[subfitness[j][0]] += (subfitness[j + 1][1][i] - subfitness[j - 1][1][i]) / (Vmax - Vmin)


def fastNonDominatedSort(fitness, cnt):
    n = []
    s = []
    i = 0
    nondominatedSet = []
    k = 0
    rank = []
    for p in fitness:
        n.append(0)
        rank.append(-1)
        s.append([])
        j = 0
        for q in fitness:
            if q is p:
                j += 1
                continue
            else:
                if cnt[i] < cnt[j]:
                    s[i].append(j)
                elif cnt[j] < cnt[i]:
                    n[i] += 1
                else:
                    if isDominated(p, q):
                        s[i].append(j)
                    elif isDominated(q, p):
                        n[i] += 1
            j += 1

        if n[i] == 0:
            nondominatedSet.append([])
            rank[i] = 0
            nondominatedSet[k].append((i, fitness[i]))
        i += 1

    while len(nondominatedSet[k]) != 0:
        nondominatedSet.append([])
        for p in nondominatedSet[k]:
            for q in s[p[0]]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = k + 1
                    nondominatedSet[k + 1].append((q, fitness[q]))
        k += 1

    for i in range(len(nondominatedSet) - 1, -1, -1):
        if nondominatedSet[i] == []:
            del nondominatedSet[i]

    return rank, nondominatedSet


def bytes2dec(bytesData, size=DEFAULTSIZE):
    decData = []
    j = 0
    for bytesArray in bytesData:
        decData.append(0)
        i = 0
        for data in bytesArray:
            decData[j] += data * 256 ** (size - 1 - i)
            i += 1
        j += 1
    return decData


def bytes2pos(bytesData, low, high, size=DEFAULTSIZE):
    dec = bytes2dec(bytesData, size)
    return low + np.array(dec) * (high - low) / (256 ** size - 1)


def bytesXor(x, y, size=DEFAULTSIZE):
    k = bytearray(size)
    for i in range(0, size):
        k[i] = x[i] ^ y[i]
    return k


def bytesAnd(x, y, size=DEFAULTSIZE):
    k = bytearray(size)
    for i in range(0, size):
        k[i] = x[i] & y[i]
    return k


def exchange(parent1, parent2, size=DEFAULTSIZE):
    k = bytesXor(parent1, parent2)
    x = bytearray(size)
    for i in range(0, size):
        x[i] = 255

    index = np.random.randint(1, 8 * size + 1)

    i = 0
    while index > 8:
        x[i] = 0
        i += 1
        index -= 8
    x[i] >>= index
    k = bytesAnd(k, x)
    offset1 = bytesXor(parent1, k)
    offset2 = bytesXor(parent2, k)

    return offset1, offset2


def rouletteWheel(fitPro):
    r1, r2 = 0, 0
    random_pro = np.random.rand()
    __sum = 0
    for i in range(0, len(fitPro)):
        __sum += fitPro[i]
        if random_pro <= __sum:
            r1 = i

    random_pro = np.random.rand()
    __sum = 0
    for i in range(0, len(fitPro)):
        __sum += fitPro[i]
        if random_pro <= __sum:
            r2 = i

    return r1, r2


def binaryTournament(rank, distance, selected):
    r1 = np.random.choice(selected)
    r2 = np.random.choice(selected)
    while r1 == r2:
        r2 = np.random.choice(selected)

    if isCrowdedDominated(r1, r2, rank, distance):
        return r1
    else:
        return r2


def checkBoundary(position, low, high):
    for i in range(0, position.shape[0]):
        for j in range(0, position.shape[1]):
            if position[i][j] < low[j]:
                position[i][j] = low[j]
            if position[i][j] > high[j]:
                position[i][j] = high[j]


def GA(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    crossoverRate = 0.5
    mutateRate = 0.01
    bitSize = DEFAULTSIZE

    # init()
    positionBit = []
    for i in range(0, N):
        positionBit.append([])
        for j in range(0, D):
            positionBit[i].append(bytearray(bitSize))
            for k in range(0, bitSize):
                positionBit[i][j][k] = np.random.randint(0, 256)

    offset = []
    for i in range(0, N):
        offset.append([])
        for j in range(0, D):
            offset[i].append(bytearray(bitSize))

    position = np.zeros([N, D])
    parents = []
    Gb = float("inf")
    t = 0
    while t < T:

        for i in range(0, N):
            position[i] = bytes2pos(positionBit[i], benchmark.low, benchmark.high)

        fitness = benchmark.getValue(position)
        Gb = min(np.min(fitness), Gb)
        # select()
        fitPro = fitness / sum(fitness)
        for i in range(0, N // 2):
            parent1, parent2 = rouletteWheel(fitPro)
            parents.append(parent1)
            parents.append(parent2)

        # crossover()
        for i in range(0, N // 2):
            offset[2 * i] = positionBit[parents[2 * i]]
            offset[2 * i + 1] = positionBit[parents[2 * i + 1]]
            crossoverRand = np.random.rand()
            if crossoverRand < crossoverRate:
                for j in range(0, D):
                    rand1 = np.random.rand()
                    if rand1 > 0.5:
                        offset[2 * i][j], offset[2 * i + 1][j] = exchange(offset[2 * i][j], offset[2 * i + 1][j])

        # mutate()
        for i in range(0, N):
            for j in range(0, D):
                for k in range(0, bitSize):
                    for l in range(0, 8):
                        rand2 = np.random.rand()
                        if rand2 < mutateRate:
                            offset[i][j][k] ^= (1 << l)

        positionBit = offset
        t += 1

    for i in range(0, N):
        position[i] = bytes2pos(positionBit[i], benchmark.low, benchmark.high)
    fitness = benchmark.getValue(position)
    Gb = min(np.min(fitness), Gb)

    return Gb


def NSGA_II(benchmark, T):
    N = benchmark.N
    D = benchmark.D
    K = benchmark.K
    crossoverRate = 0.8
    bitSize = DEFAULTSIZE
    mutateRate = 1 / (bitSize * 8)

    # init()
    positionBit = []
    for i in range(0, N):
        positionBit.append([])
        for j in range(0, D):
            positionBit[i].append(bytearray(bitSize))
            for k in range(0, bitSize):
                positionBit[i][j][k] = np.random.randint(0, 256)

    offsetBit = []
    for i in range(0, N):
        offsetBit.append([])
        for j in range(0, D):
            offsetBit[i].append(bytearray(bitSize))

    parentsBit = []
    for i in range(0, N):
        parentsBit.append([])
        for j in range(0, D):
            parentsBit[i].append(bytearray(bitSize))

    currentParentsBit = []
    for i in range(0, N):
        currentParentsBit.append([])
        for j in range(0, D):
            currentParentsBit[i].append(bytearray(bitSize))

    position = np.zeros([N, D])
    parentsFit = np.zeros([N, K])
    storedCnt = np.zeros(N)

    Gb = []
    t = 0
    while t < T:
        print(t)
        selected = []
        parents = []
        for i in range(0, N):
            position[i] = bytes2pos(positionBit[i], benchmark.low, benchmark.high)

        checkBoundary(position, benchmark.low, benchmark.high)
        fitness, cnt = benchmark.getValue(position)

        if t != 0:
            fitness = np.row_stack((fitness, parentsFit))
            cnt = np.append(cnt, storedCnt)

        rank, nondominatedSet = fastNonDominatedSort(fitness, cnt)
        distance = np.zeros(len(rank))
        for subfitness in nondominatedSet:

            crowdingDistanceAssignment(subfitness, distance)
            if len(selected) + len(subfitness) <= N:
                for i in range(0, len(subfitness)):
                    selected.append(subfitness[i][0])
            else:
                temp = []
                for i in range(0, len(subfitness)):
                    temp.append((subfitness[i][0], distance[subfitness[i][0]]))
                temp.sort(key=lambda x: x[1], reverse=True)
                for i in range(0, N - len(selected)):
                    selected.append(temp[i][0])
                break

        # select()
        for i in range(0, N):
            parent = binaryTournament(rank, distance, selected)
            parents.append(parent)

        # crossover()
        for i in range(0, N // 2):

            offsetBit[2 * i] = parentsBit[parents[2 * i] - N] if parents[2 * i] > N - 1 else positionBit[parents[2 * i]]
            offsetBit[2 * i + 1] = parentsBit[parents[2 * i + 1] - N] if parents[2 * i + 1] > N - 1 else positionBit[
                parents[2 * i + 1]]

            currentParentsBit[2 * i] = copy.deepcopy(offsetBit[2 * i])
            currentParentsBit[2 * i + 1] = copy.deepcopy(offsetBit[2 * i + 1])
            parentsFit[2 * i] = copy.deepcopy(fitness[parents[2 * i]])
            parentsFit[2 * i + 1] = copy.deepcopy(fitness[parents[2 * i + 1]])
            storedCnt[2 * i] = copy.deepcopy(cnt[parents[2 * i]])
            storedCnt[2 * i + 1] = copy.deepcopy(cnt[parents[2 * i + 1]])

            crossoverRand = np.random.rand()
            if crossoverRand < crossoverRate:
                for j in range(0, D):
                    rand1 = np.random.rand()
                    if rand1 > 0.5:
                        offsetBit[2 * i][j], offsetBit[2 * i + 1][j] = exchange(offsetBit[2 * i][j],
                                                                                offsetBit[2 * i + 1][j])

        # mutate()
        for i in range(0, N):
            for j in range(0, D):
                for k in range(0, bitSize):
                    for l in range(0, 8):
                        rand2 = np.random.rand()
                        if rand2 < mutateRate:
                            offsetBit[i][j][k] ^= (1 << l)

        positionBit = copy.deepcopy(offsetBit)
        parentsBit = copy.deepcopy(currentParentsBit)
        t += 1

    for item in nondominatedSet[0]:
        Gb.append(item[1])

    return Gb

