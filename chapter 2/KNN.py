from numpy import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import operator
from os import listdir

# ***************************************************************
# START: SIMPLE KNN


def createDataSet():
    group = array([[1.1, 1.0], [1.0, 1.0], [0.0, 0.0],[0.1,0.0]])
    lable = ['A','A','B','B']
    return group, lable


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqdiffMat = diffMat ** 2
    sumsqdiffMat = sqdiffMat.sum(axis = 1)
    distance = sumsqdiffMat ** 0.5
    sortedDistance = distance.argsort()

    # init a dictionary
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistance[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def simpleKNN():
    group, label = createDataSet()
    inX = [0.0,0.0]
    k = 3
    return classify(inX, group, label, k)


# END OF SIMPLE KNN
# ***************************************************************

# ***************************************************************
# START: DATING KNN


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    lineNumber = len(arrayOLines)
    returnMat = zeros((lineNumber,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1

    return returnMat, classLabelVector


def numLable(classLableVector):
    numDatingLabel = []
    for item in classLableVector:
        if item == 'largeDoses':
            numDatingLabel.append('3')
        elif item == 'smallDoses':
            numDatingLabel.append('2')
        else:
            numDatingLabel.append('1')
    return numDatingLabel


def datingKNN():
    filename = 'datingTestSet.txt'

    datingDataMat, datingLable = file2matrix(filename)
    datingLableNum = numLable(datingLable)
    # drawDatingFig(datingDataMat, datingLableNum)
    return


def drawDatingFig(datingDataMat, datingLableNum):
    # X, Y, Lable = datingDataMat[:,1],datingDataMat[:,2], array(datingLableNum).astype(float)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2], s=20.0 * array(datingLableNum).astype(float), c=1.0 * array(datingLableNum).astype(float))

    # draw legend
    classes = ['largeDoses', 'smallDoses', 'didntLike']
    class_colours = ['g', 'r', 'b']
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    plt.legend(recs, classes, loc=4)
    plt.xlabel('video games')
    plt.ylabel('ice cream')
    plt.title('dating KNN')
    ax.legend()
    plt.show()


def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal

    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVal,(m,1))
    normDataSet /= tile(ranges, (m,1))
    return normDataSet, ranges, minVal


def datingClassTest():
    hoRatio = 0.10
    filename = 'datingTestSet.txt'
    datingDataMat, datingLable = file2matrix(filename)
    NormDataMat, ranges, minVal = autoNorm(datingDataMat)
    m = NormDataMat.shape[0]
    count = 0.0
    TrainNum = int(hoRatio * m)
    for i in range(TrainNum):
        classifierResult = classify(NormDataMat[i,:], NormDataMat[TrainNum:m,:], datingLable[TrainNum:m], 3)
        if not classifierResult == datingLable[i]:
            count += 1

    print("error rate = ", count / m)
    return


def classifyPerson():
    filename = 'datingTestSet.txt'

    result_list = ['not at all','in small doses', 'in large doses']
    video = float(raw_input(\
        "percentage of time spent playing video games?"))
    fliemile = float(raw_input(\
        "frequent filier miles earned per year?"))
    icecream = float(raw_input(\
        "liters of ice cream consumed per year?"))
    datingDataMat, datingLables = file2matrix(filename)
    datingLableNum = numLable(datingLables)
    NormMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([video, fliemile, icecream])
    classfierResult = classify((inArr-minVals)/ranges, NormMat, datingLableNum, 3)

    print("You will probably like this person:",\
          result_list[int(classfierResult) - 1])

# END OF DATING KNN
# ***************************************************************


def img2vector(filename):
    returnVector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector


def handwritingClassTest():
    hwLabels = []
    TrainingFileList = listdir('trainingDigits')
    m = len(TrainingFileList)
    trainingMat = zeros((m,1024))

    for i in range(m):
        fileNameStr = TrainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/'+fileNameStr)

    testFileList = listdir('testDigits')

    errorCount = 0.0

    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/'+fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels,3)
        if not classifierResult == hwLabels[i]:
            errorCount += 1.0

    print('\nerror rate = ', float(errorCount/mTest))

if __name__ == '__main__':
    # print(simpleKNN())

    # datingClassTest()

    # classifyPerson()

    handwritingClassTest()