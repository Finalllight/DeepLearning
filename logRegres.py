from math import exp

import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    for num in range(len(inX)):
        inX[num][0]=1.0 / (1 + exp(-1*inX[num][0]))
    return inX


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.array(dataMatIn)
    labelMat = np.array(classLabels)
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix @ weights)
        error=[]
        for i in range(100):
            error.append(labelMat[i] - h[i][0])
        error = np.array(error)
        # error = (labelMat - h)
        l= alpha * dataMatrix.transpose() @ error
        for i in range(3):
            weights[i][0] = weights[i][0] +l[i]
    return weights

dataArr,labe=loadDataSet()
ans=gradAscent(dataArr,labe)
print(ans)
# dataMatrix = np.array(dataArr)
# labelMat = np.array(labe)
# m, n = np.shape(dataMatrix)
# alpha = 0.001
# maxCycles = 500
# weights = np.ones((n, 1))
# h = sigmoid(dataMatrix @ weights)
# error = []
# for i in range(100):
#     error.append(labelMat[i] - h[i][0])
# error = np.array(error)
# print(dataMatrix.transpose().shape)
# print(error.shape)
# l =dataMatrix.transpose() @ error
#
# print(weights.shape)



