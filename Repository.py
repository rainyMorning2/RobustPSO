import numpy as np
import sys

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
    __storedFit = []
    __storedPos = []

    def __init__(self, maxSize, divisions):
        self.maxSize = maxSize
        self.divisions = divisions

    def size(self):
        return len(self.__storedFit)

    def get(self, index):
        if index < len(self.__storedFit):
            return self.__storedFit[index]
        else:
            raise IndexError

    def info(self):
        print(self.size())
        print(self.__storedPos)
        print(self.__storedFit)

    def __genGrid(self):
        maxFit = np.max(self.__storedFit, axis=0)
        minFit = np.min(self.__storedFit, axis=0)
        deta = (maxFit - minFit)/self.divisions
        id = np.floor((self.__storedFit - minFit)/deta).astype(np.int)
        id[id == self.divisions] = self.divisions - 1
        grid = list([])

        # means position , particle indexes which are in current position , probability (equals to n/size )
        grid.append([id[0], [0], 0])

        for i in range(1, len(id)):
            for j in range(0, len(grid)):
                if np.all(grid[j][0] == id[i]):
                    grid[j][1].append(i)
                else:
                    grid.append([id[i], [i], 0])
        

        return id, grid

    def insert(self, itemPos, itemFit):
        if self.size() == 0:
            self.__storedFit.append(itemFit)
            self.__storedPos.append(itemPos)
        else:
            for i in range(self.size() - 1, -1, -1):
                if isDominated(self.__storedFit[i], itemFit):
                    return
                elif isDominated(itemFit, self.__storedFit[i]):
                    self.__storedFit.remove(self.__storedFit[i])
                    self.__storedPos.remove(self.__storedPos[i])

            if self.size() < self.maxSize:
                self.__storedFit.append(itemFit)
                self.__storedPos.append(itemPos)
            else:
                # using Grid to eliminate items
                id , grid = self.__genGrid()
                print(grid)



    def remove(self, index):
        if index < len(self.__storedFit):
            self.__storedFit.remove(self.__storedFit[index])
            self.__storedPos.remove(self.__storedPos[index])
        else:
            raise IndexError


if __name__ == '__main__':
    repo = Repository(2, 30)
    repo.info()
    a = [2, 3]
    b = [2, 2]
    c = [3, 5]
    d = [4, 7]
    e = [1, 0]
    af = [20, 40]
    bf = [30, 50]
    cf = [25, 20]
    df = [50, 10]
    ef = [5, 5]
    repo.insert(a, af)
    repo.info()
    repo.insert(b, bf)
    repo.info()
    repo.insert(c, cf)
    repo.info()
    repo.insert(d, df)
    repo.info()


