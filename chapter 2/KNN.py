from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import operator


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
    filename = './MLInActionCode-master/Ch02/datingTestSet.txt'

    datingDataMat, datingLable = file2matrix(filename)
    datingLableNum = numLable(datingLable)
    drawDatingFig(datingDataMat, datingLableNum)
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


if __name__ == '__main__':
    # print(simpleKNN())

    datingKNN()

