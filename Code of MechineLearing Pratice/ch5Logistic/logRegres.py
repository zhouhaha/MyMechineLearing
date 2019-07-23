#coding=utf-8
from numpy import *

#导入txt文档，然后返回一个包含[1.0，X1，X2]的二维数组dataMat，以及其对应的标签labelMat
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split() #逐行读入并切分，每行的前两个值为X1，X2
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#X0设为1.0，保存X1，X2,添加了一个数组
        labelMat.append(int(lineArr[2])) #每行第三个值对应类别标签   添加对应的标签
    print(dataMat,labelMat)
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#输入一个特征参数矩阵dataMatIn，一个类别标签向量classLabels，返回迭代跟新之后的权重值w对应的矩阵weights（为一个3*1的一维矩阵）
def gradAscent(dataMatIn,classLabels):#100,3 matrix   #1dataMatIn为一个100*3的数据矩阵，classLabels为其对应的标签
    dataMatrix=mat(dataMatIn) #change to numpy matrix ,different features for col &sample for row
    labelMat=mat(classLabels).transpose()    #将行向量改为列向量
    m,n=shape(dataMatrix)  #返回输入的特征矩阵的行数、列数
    #parameter for train
    alpha=0.001 #step length步长
    maxCycles=500#iteration num  迭代次数
    weights=ones((n,1))    #定义一个n行1列每个单元值为1的的矩阵，这里shape（）已经获取到n值
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights) #h is a vector  h是一个100*1的向量，这有是矩阵的乘法运算
        error=(labelMat-h) #compute the difference between real type and predict type  得到实际标签值与计算的标签值之间的误差
        weights=weights+alpha*dataMatrix.transpose()*error  #梯度上升更新权重
    #下面是我自己加的，为的是列表与列表之间类型能够对应，不然运行不了
    newweights = []
    weights = weights.tolist()
    print(weights)
    for i in weights:
        newweights.extend(i)
    print(newweights)
    return newweights #return the best parameter

#输出权重list（[w0,w1,w2]）,导入txt文档的数据，然后根据标签将数据分类，标签为0、为1各分一类，得到特征的坐标（x，y），
# 然后以z = w[0]*dataMat[0]+w[1]*dataMat[1]+w[2]*dataMat[2] = 0作为分隔标签为0和标签为1的分割线，并将坐标和分割线画出来
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)  #将二维的list变为（[]）x形式
    n = shape(dataArr)[0]  #获取返回的数据的行数
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):   #这里的作用是将标签为0和为1对应的数据特征样本分开保存，第一列保存进X的list，第二列特征保存为y的list
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')   #画散点图
    ax.scatter(xcord2,ycord2,s=30,c='blue')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1] * x) / weights[2] #在前面导入数据的时候定义每一行的X[0]=1.0，因z = w[0]*dataMat[0]+w[1]*dataMat[1]+w[2]*dataMat[2]，令z=0，可以得出dataMat[1]与dataMat[2]之间的关系，即X与Y之间的关系
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()

#随机梯度上升法，可以进行增量是学习权重weights，当新来的样本，直接可以在原有的weights上面加上训练参数进行训练
# 返回训练好的权重参数向量weights，但是其计算过程中的变量h和误差error都为数值，不是向量
def stocGradAscent0(dataMatrix, classLabels):
    if type(dataMatrix) != 'numpy.ndarray':
        dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))  #为一个数值
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#这是添加了可以重复训练多次的随机梯度上升，返回的是权重weights列表
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    if type(dataMatrix) != 'numpy.ndarray':
        dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))  # python3 change: dataIndex=range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 设置了不长alpha随着迭代次数和数据样本数的变化关系
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])  # 将上述的那个值删除
    return weights



#test logistic
#分类
def classifyVector(intX ,weights):
    prob = sigmoid(sum(intX * weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
#对horseColic数据进行训练和测试，返回其测试的误差率
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(lineArr)
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is:%f" % errorRate)
    return errorRate

#求多次训练和测试之后的误差平均误差率
def multiTest():
    numTests=6;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iteration the average error rate is:%f" % (numTests,errorSum/float(numTests)))


# if __name__ == "__main__":
#     a, b = loadDataSet()
#     c = gradAscent(a, b)
#     plotBestFit(c)
    # multiTest()