import benchmarks
import PSO
import Robust_PSO
import openpyxl
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

def run():

    # number of particles
    N = 50
    # dimensions
    D = 2
    # iteration times
    T = 50
    # number of random particles
    m = 4
    # bias
    e = 0.35
    # number of cal
    Time = 100

    benchmark = benchmarks.testRobust(N, D)

    '算法数'
    num = 5

    result = []
    names = []

    for index in range(0, 1):
        # index of benchmarks

        wbk = openpyxl.Workbook()

        sheets = [wbk.create_sheet('data', 0), wbk.create_sheet('PSO', 1), wbk.create_sheet('RobustPSO', 2),
                  wbk.create_sheet('RingPSO', 3), wbk.create_sheet('SFPSO', 4), wbk.create_sheet("SWPSO", 5)]

        # name = str(type(benchmarks[index]))[19:-2]
        name = "testBenchmark"
        times = 0

        while times < Time:

            data = []
            data.append(PSO.PSO(benchmark, T))
            data.append(Robust_PSO.PSO(benchmark, T, m, e))
            data.append(Robust_PSO.RPSO(benchmark, T, m, e))
            data.append(Robust_PSO.SFPSO(benchmark, T, m, e))
            data.append(Robust_PSO.SWPSO(benchmark, T, m, e))

            print(name + ' ' + str(times))
            for i in range(0, num):
                for j in range(0, data[i].__len__()):
                    sheets[i + 1].cell(row=j + 2, column=times + 2).value = data[i][j]

            times += 1

        avg = []
        selected = []

        for i in range(0, num):

            avg.append([])
            selected.append([])

            for j in range(0, Time):
                sheets[i + 1].cell(row=1, column=j + 2).value = j + 1

            for j in range(0, data[i].__len__()):
                sheets[i + 1].cell(row=j + 2, column=1).value = (j + 1) * N

            for col in range(2, sheets[i + 1].max_column + 1):
                # if sheets[i + 1].cell(row=sheets[i + 1].max_row, column=col).value < benchmarks[index].goal:
                selected[i].append(col)

            for row in range(2, sheets[i + 1].max_row + 1):
                avg[i].append(0)
                for col in selected[i]:
                    avg[i][row - 2] += sheets[i + 1].cell(row=row, column=col).value
                if selected[i].__len__() != 0:
                    avg[i][row - 2] /= selected[i].__len__()

        for i in range(0, num):
            for row in range(2, avg[i].__len__() + 2):
                sheets[0].cell(row=row, column=1).value = (row - 1) * N
                sheets[0].cell(row=row, column=i + 2).value = avg[i][row - 2]

        sheets[0].cell(row=1, column=2).value = "PSO"
        sheets[0].cell(row=1, column=3).value = "RobustPSO"
        sheets[0].cell(row=1, column=4).value = "RingPSO"
        sheets[0].cell(row=1, column=5).value = "SFPSO"
        sheets[0].cell(row=1, column=6).value = "SWPSO"

        # '成功率'
        # sheets[0].cell(row=1, column=9).value = selected[0].__len__() / Time
        # sheets[0].cell(row=1, column=10).value = selected[1].__len__() / Time
        # sheets[0].cell(row=1, column=11).value = selected[2].__len__() / Time
        # sheets[0].cell(row=1, column=12).value = selected[3].__len__() / Time
        # sheets[0].cell(row=1, column=13).value = selected[4].__len__() / Time

        wbk.save('.//result//' + name + '.xlsx')

        # bingoRate.append([])
        # result.append([])
        # names.append(name)
        # for i in range(0, num):
        #     bingoRate[index].append(selected[i].__len__() / Time)
        #     result[index].append(avg[i][avg[i].__len__() - 1])

    # wbk = openpyxl.Workbook()
    # sheet = wbk.create_sheet('result', 0)
    #
    # for j in range(0, names.__len__()):
    #     sheet.cell(row=j + 2, column=1).value = name[j]
    #     for i in range(0, num):
    #         sheet.cell(row=j + 2, column=i + 2).value = result[j][i]
    #         sheet.cell(row=j + 2, column=i + num + 2).value = bingoRate[j][i]
    #
    # wbk.save('.//result//' + 'result' + '.xlsx')


def test():

    # number of particles
    N = 50
    # dimensions
    D = 2
    # iteration times
    T = 100
    # number of random particles
    m = 8
    # bias
    e = 0.35

    b = benchmarks.testRobust(N, D)

    t = np.zeros([1,2])
    t[0][0]=0.4
    t[0][1]=0.2
    print(b.getValue(t)+35)

    t[0][0] = 2
    t[0][1] = 1
    print(b.getValue(t)+35)

    t[0][0] = 2.8
    t[0][1] = 4.0
    print(b.getValue(t)+35)

    t[0][0] = 1
    t[0][1] = 2.5
    print(b.getValue(t) + 35)
    # PSO.PSO(b, T)
    # Robust_PSO.PSO(b, T, m, e)
    # Robust_PSO.SFPSO(b, T, m, e)
    # Robust_PSO.RPSO(b, T, m, e)
    # Robust_PSO.SWPSO(b, T, m, e)


def draw():
    div = 200
    N = div*div
    D = 2

    b = benchmarks.testRobust(N, D)
    x, y = np.linspace(-0.5, 3.1, div), np.linspace(4.3, 0, div)
    position = np.zeros([N, D])

    index1 = 0
    index2 = 0
    for i in range(0, N):
        position[i][0] = x[index1]
        position[i][1] = y[index2]
        index1 = index1 + 1
        if index1 == div:
            index1 = 0
            index2 = index2 + 1

    z = (b.getValue(position)).reshape(div, div)
    plt.imshow(z+35, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
               cmap=cm.hot, norm=LogNorm())
    plt.colorbar()
    plt.plot([0.3, 2.8, 1], [2, 4, 2.5], 'bo')
    plt.show()

if __name__ == '__main__':
    draw()