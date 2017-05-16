# -*-coding:utf-8-*-
from math import log
import operator



def calShannonEntr(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for sample in dataSet:
        currentLabel = sample[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEtro = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEtro -= prob * log(prob, 2)

    return shannonEtro


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVect in dataSet:
        if featVect[axis] == value:
            reducedFeatVec = featVect[:axis]
            reducedFeatVec.extend(featVect[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseShannonEtro = calShannonEntr(dataSet)

    bestInfoGain, bestFeatures = 0.0, -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVal= set(featList)

        newEntropy = 0.0

        for v in uniqueVal:
            subDataSet = splitDataSet(dataSet, i, v)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calShannonEntr(subDataSet)

        # 信息熵越小，说明越不乱，也就说明这个分类能带来信息，所以信息增益大
        # 别忘了信息熵返回回来的时候就是负值
        infoGain = baseShannonEtro - newEntropy

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatures = i

    return bestFeatures

# test  ---- create a small data set to test shannonEntro
def createDataSet():
    dataSet = [[1,1, 'yes'],
               [1,1, 'yes'],
               [1,0, 'no'],
               [0,1, 'no'],
               [0,1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.iteritems(),\
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # 统计类别，还剩多少label
    classList = [example[-1] for example in dataSet]
    # 如果所有label都一样，说明这个组已经分类好了
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果只剩一个features，那这个feature只需要直接投票了，少数服从多数
    if len(dataSet[0]) == 1:
        return  majorityCnt(classList)

    # 先选一个最好的
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVal = set(featValues)

    for value in uniqueVal:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(\
            dataSet,bestFeat, value), subLabels)

    return myTree


def classify_decisionTree(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify_decisionTree(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel



# def drawDecisionTree():
#     import matplotlib.pyplot as plt
#
#     decisitonNode = dict(boxstyle = "sawtooth", fc = "0.8")
#     leafNode = dict(boxstyle = "round4", fc = "0.8")
#     arrow_args = dict(arrowstyle = "<-")






if __name__ == '__main__':
    myDat, labels = createDataSet()
    # result = chooseBestFeatureToSplit(myDat)
    # print(result)

    myTree = createTree(myDat, labels)
    print(myTree)

    print(classify_decisionTree(myTree, labels, [1,0]))