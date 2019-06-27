#coding=utf-8
from numpy import *  #科学计算包numpy
import operator      #运算符模块
#k-近邻算法
#计算输入向量inX与数组dataSet的距离，并取距离最近的K个标签，计算这K个中频数最高的那个作为KNN的预测值
def classify0(inX,dataSet,labels,k):  #inX用于分类的输入向量，dataSet为训练样本数据集，labels为与dataSet对应的标签向量，k为最近邻的数目
    dataSetSize=dataSet.shape[0]   #shape读取数据训练集矩阵第一维度的长度，可以理解为行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  #tile重复数组inX，有dataSetSize行，1个dataSet列，减法计算差值
    sqDiffMat=diffMat**2 #**是幂运算的意思，这里用的欧式距离
    sqDisttances=sqDiffMat.sum(axis=1) #普通sum默认参数为axis=0为普通相加，axis=1为一行的行向量相加
    distances=sqDisttances**0.5
    sortedDistIndicies=distances.argsort() #argsort返回数值从小到大的索引值（数组索引0,1,2,3）
 #选择距离最小的k个点
    classCount={}   #建立了一个字典，字典里面的元素形式为：（A：B）
    for i in range(k):
         voteIlabel=labels[sortedDistIndicies[i]] #根据排序结果的索引值返回靠前的前k个样本所属于的标签
         classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #各个标签出现频率  get（）中的0表示如果指定的键不存在，则返回0；该语句的效果是建立了一个字典，以键值对：标签号：出现频数的形式存储于字典中
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #按照键值对中的值从大到小排序；1代表排序每个类别的频率
    #!!!!!  classCount.iteritems()修改为classCount.items()
    #sorted(iterable, cmp=None, key=None, reverse=False) --> new sorted list。
    # reverse默认升序 key关键字排序itemgetter（1）按照第一维度排序(0,1,2,3)
    return sortedClassCount[0][0]  #找出频率最高的类别，由于sort排序之后输出的元祖，不是字典了，因此这里相当于是二维数组，所以是[0][0]

#创建数据集，group为输入，AB为输出标签
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.5]]) #这里必须是array数组，不能是简单的列表，因为只有array才有shape这个属性
    labels=['A','A','B','B']
    return group,labels

#将待处理数据格式转变为分类器可以接受的格式，读取文件中的内容并存储于数组returnMat（存储特征值）和向量classLabelVector（存储数据类别）中
def file2matrix(filename):
    fr=open(filename)  #filename为打开的文件路径
    arrayOLines=fr.readlines()   #读取文件的所有行，并以list的形式把每行存储在arrayOLines里面
    numberOfLines=len(arrayOLines) #读出数据行数
    returnMat=zeros((numberOfLines,3))  #创建返回矩阵
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()  #删除字符串头尾的空白符
        listFromLine=line.split('\t') #split指定分隔符（\t为制表符）对数据切片
        returnMat[index,:]=listFromLine[0:3] #选取前3个元素（特征）存储在返回矩阵中。也可认为按行填充returnMat矩阵
        classLabelVector.append(int(listFromLine[-1]))
        #-1索引表示最后一列元素,位label信息存储在classLabelVector
        index+=1
    return returnMat,classLabelVector

#归一化特征值。归一化公式  ：（当前值-最小值）/range，返回归一化数数值矩阵和范围参数以及最小参数三个数据
def autoNorm(dataSet):
    minVals=dataSet.min(0) #存放每一列中的最小值，参数0使得可以从列中选取最小值，而不是当前行，最后得到的是一行数据
    maxVals=dataSet.max(0) #存放每一列中的最大值，最后得到的是一行数据，
    ranges = maxVals - minVals
    normDataSet=zeros(shape(dataSet))  #初始化归一化矩阵为读取的dataSet
    m=dataSet.shape[0]  #m保存第一行，m为dataSet的行数
    # 特征矩阵是3x1000，min max range是1x3 因此采用tile将变量内容复制成输入矩阵同大小
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))  #normDataSet是一个1000*3的矩阵
    return normDataSet, ranges, minVals

#测试约会网站分类结果代码，打印出测试集中的测试结果与真实结果，并且打印出最后的测试集的错误率
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  #从数据集中拿出10%作为测试的数据集
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)#对于KNN而言，拿出10%作为测试集，则剩下的90%为训练集，因此是需要全部作为已知数据集拿来判断测试集
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    #print(errorCount)

#完整的约会网站预测：给定一个人，判断时候适合约会
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video games?"))#书中raw_input在python3中修改为input（）
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')#原书没有2，datingLabels向量是与datingDataMat矩阵的行数相同，并一一对应的
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:", resultList[classifierResult-1])

import os, sys
#按照每一行读取txt文件中的32*32像素所代表的0/1值，并将其转为一个1*1024的数组里面，返回所存储的1*1024的数组
def img2vector(filename):
    returnVect=zeros((1,1024))#每个手写识别为32x32大小的二进制图像矩阵 转换为1x1024 numpy向量数组returnVect
    fr=open(filename)#打开指定文件
    for i in range(32):#循环读出前32行
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])#将每行的32个字符值存储在numpy数组中
    return returnVect

#文字识别的测试算法
#本函数需要确保将from os import listdir写入文件的起始部分，这段代码的主要功能是从os模块中导入函数listdir，它可以列出给定目录的文件名。
#训练集先获取文件夹下面所有的文件名，根据文件名得到每个文件的类别作为labels存储于一个一维数组中，再读取文件内容转为1*1024的一维矩阵，组合这些训练集中的一维矩阵，构成一个m*1024的二维矩阵。
#测试集获取文件名，根据文件名得到文件的真实类别，根据文件内容导入classify0（）分辨函数里进行判断，输出对应的类别，将推导的类别与真实类别进行比对，如果不同，则错误+1，最后输出错误率
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=os.listdir('trainingDigits')#修改 import os 这里加上os.，列出文件夹trainingDigits中所有文件的文件名，并存储于trainingFileList中
    m=len(trainingFileList) #返回训练集的文件个数
    trainingMat=zeros((m,1024)) #定义文件数x每个向量的训练集
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]#解析文件，fileNameStr.split('.')将文件名按‘.’分隔，分隔之后的结果是一个list，现在取list中的第一个元素
        classNumStr=int(fileStr.split('_')[0])#解析文件名，由于文件名的命名为0_1.txt的形式，分隔之后classNumStr即为该txt文件所存储的数字类别是多少
        hwLabels.append(classNumStr)#存储类别
        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr) #访问第i个文件内的数据，并将其转为1*1024的一维数组的格式
    #测试数据集
    testFileList=os.listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])#从文件名中分离出数字作为基准
        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)#访问第i个文件内的测试数据，不存储类 直接测试
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d,the real answer is: %d" %(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1.0
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total rate is:%f"% (errorCount/float(mTest)))









