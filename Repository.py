import numpy as np
import matplotlib.pyplot as plt

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


class Repository(object):

    def __init__(self, maxSize, divisions=1):
        self.maxSize = maxSize
        self.divisions = divisions
        self.__storedFit = []
        self.__storedPos = []
        self.__x = 10
        self.__grid = list([])

    def __rouletteWheel(self):
        random_pro = np.random.rand()
        __sum = 0
        for i in range(0, len(self.__grid)):
            __sum += self.__grid[i][2]
            if random_pro <= __sum:
                return i

    def __genDeletePro(self):
        for item in self.__grid:
            item[2] = len(item[1]) / self.size()

    def __genSelectPro(self):
        __sum = 0
        for item in self.__grid:
            item[2] = self.__x / len(item[1])
            __sum += self.__x / len(item[1])

        # Normalize
        for item in self.__grid:
            item[2] /= __sum

    def updateGrid(self):
        maxFit = np.max(self.__storedFit, axis=0)
        minFit = np.min(self.__storedFit, axis=0)
        deta = (maxFit - minFit) / self.divisions
        id = np.floor((self.__storedFit - minFit) / deta).astype(np.int)
        id[id == self.divisions] = self.divisions - 1
        # means position , particle indexes which are in current position , probability (select or delete)
        self.__grid = []
        self.__grid.append([id[0], [0], 0])
        for i in range(1, self.size()):
            for j in range(0, len(self.__grid)):
                if np.all(self.__grid[j][0] == id[i]):
                    self.__grid[j][1].append(i)
                else:
                    self.__grid.append([id[i], [i], 0])
                break

        while self.size() > self.maxSize:
            self.__genDeletePro()
            pos = self.__rouletteWheel()
            index = np.random.choice(self.__grid[pos][1])
            del self.__storedPos[index]
            del self.__storedFit[index]
            self.__grid[pos][1].remove(index)
            if len(self.__grid[pos][1]) == 0:
                del self.__grid[pos]
            for item in self.__grid:
                for i in range(0, len(item[1])):
                    if item[1][i] > index:
                        item[1][i] -= 1

    def insert(self, itemFit, itemPos=None):
        for i in range(self.size() - 1, -1, -1):
            if np.all(abs(self.__storedFit[i] - itemFit) < 1e-4) or isDominated(self.__storedFit[i], itemFit):
                return
            elif isDominated(itemFit, self.__storedFit[i]):
                del self.__storedFit[i]
                del self.__storedPos[i]

        self.__storedFit.append(itemFit)
        self.__storedPos.append(itemPos)

    def size(self):
        return len(self.__storedFit)

    def get(self):
        self.__genSelectPro()
        return self.__storedPos[np.random.choice(self.__grid[self.__rouletteWheel()][1])]

    def getAll(self):
        return self.__storedFit

    def info(self):
        print(self.size())
        print(self.__storedPos)
        print(self.__storedFit)
        print(self.__grid)

    def clear(self):
        self.__storedPos = []
        self.__storedFit = []
        self.__grid = list([])

    def plot(self):
        true = np.loadtxt("Kursawe_fun.dat")
        plt.plot(true[:, 0], true[:, 1], "r.", ms=2)
        plt.plot(np.array(self.__storedFit)[:, 0], np.array(self.__storedFit)[:, 1], "+", ms=5)
        plt.xlabel('$f_{1}$')
        plt.ylabel('$f_{2}$')
        plt.show()

    def save(self, filename):
        np.savetxt(filename, self.__storedFit)


if __name__ == '__main__':

    repo = Repository(3, 30)

    repo.plot()

    # a = [2, 3]
    # b = [2, 2]
    # c = [3, 5]
    # d = [4, 7]
    # e = [1, 0]
    # af = [20, 40]
    # bf = [30, 35]
    # cf = [25, 20]
    # df = [50, 10]
    # ef = [5, 5]
    # repo.insert(a, af)
    # repo.info()
    # repo.insert(b, bf)
    # repo.info()
    # repo.insert(c, cf)
    # repo.info()
    # repo.insert(d, df)
    #
    # repo.updateGrid()
    # repo.info()
    # print(repo.get())



