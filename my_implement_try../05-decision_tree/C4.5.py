#coding:utf-8
import pandas as pd
import numpy as np
from math import log


def calcEmpiricalEntropy(dataset):
    """
    :param dataset: 数据集
    :return: Empirical Entropy 计算给定数据集的经验熵(H(D)) 见公式(5-7)
    """
    numEntries = dataset.shape[0]  
    labelCounts = {} 
    cols = dataset.columns.tolist() 
    classlabel = dataset[cols[-1]].tolist()  # 将数组或者矩阵转换成列表
    for label in classlabel:
        if label not in labelCounts.keys():
            labelCounts[label] = 1
        else:
            labelCounts[label] += 1

    empiricalEntropy = 0.0
    for _, value in labelCounts.items():
        prob = value/numEntries
        empiricalEntropy -= prob*log(prob, 2)

    return empiricalEntropy


def splitDataSet(dataset, axis, value):
    """
    :param dataset: 数据集
    :param axis: 所占列
    :param value: 决策树的分支, 等于 value 的值
    :return: 按照给定维度上(axis)的特征的具体取值(value)划分好的子集
    """
    cols = dataset.columns.tolist()
    axisFeat = dataset[axis].tolist()
    #print("axisFeat: ", axisFeat)
    # 更新数据集
    # 去除当前特征值所在的列
    retDataSet = pd.concat( [dataset[featVec] for featVec in cols if featVec != axis] , axis=1)
    
    # 删除与当前特征值不等的行
    i = 0
    dropIndex = [] #删除项的索引集
    for featVec in axisFeat:
        if featVec != value:
            dropIndex.append(i)
        i += 1
        
    newDataSet = retDataSet.drop(dropIndex)
    
    return newDataSet.reset_index(drop=True)


def chooseMaxInfoGainRatioFeature(dataset):
    """
    :param dataset:
    :return: 选择最好的数据集划分特征并返回,式子(5-7),(5-8)
    """
    numFeatures = dataset.shape[1] - 1  # 最后一列是结果
    HD = calcEmpiricalEntropy(dataset)
    #print("HD: ", HD)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    cols = dataset.columns.tolist()
    
    for i in range(numFeatures):
        equalVals = set(dataset[cols[i]].tolist())  # 这些特征的具体取值范围
        empirCondEntropy = 0.0
        for value in equalVals:  # i--> n 对特征的取值进行求经验熵的和 第一个求和号
            # 函数 splitDataSet() 获取由特征不同取值划分的数据集
            subDataSet = splitDataSet(dataset, cols[i], value)
            # print("subDataSet: ", subDataSet)
            # |Di| : subDataSet.shape[0] 
            # |D| : dataset.shape[0]
            prob = subDataSet.shape[0] / dataset.shape[0]
            empirCondEntropy += prob * calcEmpiricalEntropy(subDataSet)
        infoGainRatio = (HD - empirCondEntropy)/HD
        # print(cols[i], infoGain)
        if infoGainRatio > bestInfoGainRatio:
            bestInfoGainRatio = infoGainRatio
            bestFeature = cols[i]
    return bestFeature, bestInfoGainRatio


def majorityVote(classList):
    """

    :param classList: 分类类别列表, 数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    :return: 节点的分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedValue = sorted(classCount.items(), key=lambda item:item[1])
    return sortedValue[-1][0]


def createTree(dataset, delt = 0.001):
    """
    :param dataset:
    :return: 返回递归构建的决策树
    """
    cols = dataset.columns.tolist()[:-1]
    classList = dataset[dataset.columns.tolist()[-1]].tolist()


    # 终止条件
    # 若数据集中所有实例属于同一类Ck，则为单节点树，并将Ck作为该节点的类标记
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 若特征集为空集，则为单节点树，并将数据集中实例数最大的类Ck作为该节点的类标记
    if len(cols) == 0:
        return majorityVote(classList)
    
    
    
    print('特征集和类别:', dataset.columns.tolist())
    bestFeature, bestInfoGainRatio = chooseMaxInfoGainRatioFeature(dataset)
    if bestInfoGainRatio < delt:
        return majorityVote(classList)
    print('bestFeture:', bestFeature)

    decisionTree = {bestFeature: {}}

    # 得到列表包括节点所有的属性值
    featValues = set(dataset[bestFeature])
    for value in featValues:
        decisionTree[bestFeature][value] = createTree( splitDataSet(dataset, bestFeature, value) )
    return decisionTree


if __name__ == '__main__':
    dataset = pd.read_csv("data.csv")
    #dataset = pd.read_csv("./Iris.csv")
    DeciTree = createTree(dataset)
    print(DeciTree)

