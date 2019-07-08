import numpy as np


def GD(paretoSet, nondominatedSet):
    n = nondominatedSet.shape[0]
    d = 0
    for item in nondominatedSet:
        d += min(np.linalg.norm(item-paretoSet, ord=2, axis=1)**2)

    return (d**0.5)/n


def SP(paretoSet, nondominatedSet):
    n = nondominatedSet.shape[0]
    d = np.zeros(n)
    i = 0
    for item in nondominatedSet:
        d[i] = min(np.linalg.norm(item - paretoSet, ord=1, axis=1) ** 2)
        i += 1
    avg = sum(d)/n

    return (sum((avg-d)**2)/(n-1))**0.5


def ER(paretoSet, nondominatedSet):
    n = nondominatedSet.shape[0]
    d = np.zeros(n)
    i = 0
    for item in nondominatedSet:
        d[i] = min(np.linalg.norm(item - paretoSet, ord=2, axis=1) ** 2)
        if d[i] < 1e-4:
            d[i] = 0
        else:
            d[i] = 1
        i += 1

    return sum(d)/n


if __name__ == '__main__':
    fucName = "Kita"
    algName = "MOPSO"
    true = np.loadtxt(".//result//MOfunc//" + fucName + "_fun.dat")
    file = np.loadtxt(".//result//" + fucName + '-' + algName + ".txt")
    print(SP(0-true, file))