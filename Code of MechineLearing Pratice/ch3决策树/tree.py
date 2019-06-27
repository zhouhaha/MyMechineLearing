#coding=utf-8
from math import log
import operator
import matplotlib.pyplot as plt
#简单鱼分类
def createDataSet():
    dataSet = [['青年','否','否','一般','否'],
               ['青年','否','否','好','否'],
               ['青年','是','否','好','是'],
               ['青年','是','是','一般','是'],
               ['青年','否','否','一般','否'],
               ['中年','否','否','一般','否'],
               ['中年','否','否','好','否'],
               ['中年','是','是','好','是'],
               ['中年','否','是','非常好','是'],
               ['中年','否','是','非常好','是'],
               ['老年','否','是','非常好','是'],
               ['老年','否','是','好','是'],
               ['老年','是','否','好','是'],
               ['老年','是','否','非常好','是'],
               ['老年','否','否','一般','否']]
    labels = ['年龄','有工作','有自己的房子','信贷情况']
    #change to discrete values
    return dataSet, labels

#计算香农熵，度量数据集无序程度
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)  #计算数据集的长度
    labelCounts={}
    for fearVec in dataSet:  #建立数据的“类别：个数”的键值对，并存储于labelCounts中。取最后一列键值 记录当前类别出现次数,
         currentLabel=fearVec[-1] #得到dataSet的最后一列的数据，为yes或者no的那个
         # labelCounts[currentLabel] = labelCounts.get(currentLabel,0)+1  此句与下面的if功能相同
         if currentLabel not in labelCounts.keys():
             labelCounts[currentLabel] = 0
         labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries #该类别的概率
        shannonEnt-=prob*log(prob,2) #计算香农熵
    return shannonEnt

#划分数据集，对于给定特征axis，返回第axis+1个特征为value的数组集合（为一个二维数组）
def splitDataSet(dataSet, axis, value):#三个参数分别为待划分的数据集 数据集特征 需要返回的特征值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
             reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
             reducedFeatVec.extend(featVec[axis + 1:])#extend方法是讲添加元素融入集合
             retDataSet.append(reducedFeatVec)#append将添加元素作为一个元素加入
    return retDataSet

# 遍历每个特征，根据选取的特征划分数据集，计算每个特征的信息增益，选取信息增益最大的特征作为最好的特征，用于数据集划分，返回最好特征
#对数据集dataSet的要求是dataSet的每一个样本的特征必须对齐，二是dataSet最后一列是样本对应的输出类别标签
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1   #numFeatures为特征的个数
    baseEntropy=calcShannonEnt(dataSet)  #计算整个数据集的信息熵
    bestInfoGain=0.0; bestFeature=-1  #后面的代码有说明，一个是最好的信息增益，另一个是最好的特征的索引值
    for i in range(numFeatures):  #遍历数据集中所有特征
        featList=[example[i] for example in dataSet] #使用列表推导来创建新的列表，获取数据集第i+1列的数据并存放于列表featList中
        uniqueVals=set(featList) #将list类型转为set类型，目的在于将list多个重复的值变为只有不同的值，例如[1,1,2,3]-->(1,2,3)
        newEntropy=0.0
        for value in uniqueVals:#遍历当前特征中的所有唯一属性值
            subDataSet=splitDataSet(dataSet,i,value)#对第i+1个特征划分数据集，返回特征为value的数组集合
            prob=len(subDataSet)/float(len(dataSet))  #得到第i+1个特征的类别分类，计算特征为value的数组集合所占比例
            newEntropy+=prob*calcShannonEnt(subDataSet)  #计算特征i+1对应的条件信息熵
        infoGain=baseEntropy-newEntropy  #得到特征‘i+1’的信息增益
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
        return bestFeature

#确定决策树叶子节点的分类类别：一列中个数最多的类别
def majorityCnt(classList):  #输入参数classList是一个列表类型
    classCount={}
    for vote in classList:    #遍历classlist，统计classlist中不同类别的个数，并按键值对的方式存储于classCount中
        if vote not in classCount.keys(): classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  ##按照键值对中的值从大到小排序；1代表排序每个类别的频率；返回形如[(),()]
    return sortedClassCount[0][0]

#创建决策树，为最主要的函数，调用了其他函数
def createTree(dataSet,labels):
    hhahah = []   #这里新建了一个列表，用于替换labels在该函数，因为labels会在本函数中删除，导致后面与真正的类别不同，这里将labels赋值给了hhahah
    for i in labels:
        hhahah.append(i)
    classList=[example[-1] for example in dataSet]  #获得数据集dataSet中的标签，并按顺序存储于classList中
    if classList.count(classList[0]) == len(classList): #判断classList中是不是只包含一类
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0])==1:  #使用完了所有的特征，最后只剩下类别那一栏，所以是等于1
        return majorityCnt(classList)#返回出现次数最多的类别
    #创建树
    bestFeat=chooseBestFeatureToSplit(dataSet)#将选取的最好特征放在bestFeat中
    bestFeatLabel=hhahah[bestFeat]   #特征标签
    myTree={bestFeatLabel:{}}      #使用特征标签创建树
    del(hhahah[bestFeat])  #del用于list列表操作，删除一个或者连续几个元素
    featValues=[example[bestFeat] for example in dataSet] #选取数据集dataSet中的特征为bestFeat+1的那一列
    uniqueVals=set(featValues)  #清除多余项，只剩类型
    for value in uniqueVals:
        subLabels = hhahah[:]  # 使用新变量代替原有列表，新的列表删除了选取的特征值那一列
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#使用决策树来分类
def classify(inputTree,featLabels,testVec):
    #python3.X
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    # python3.X
    secondDict=inputTree[firstStr]  #baocun在secondDict中
    #print(secondDict)
    featIndex=featLabels.index(firstStr)  #建立索引
    #print(featIndex)
    for key in secondDict.keys():
        if testVec[featIndex]==key: #若该特征值等于当前key，yes往下走
            if type(secondDict[key]).__name__=='dict':# 若为树结构
                classLabel=classify(secondDict[key],featLabels,testVec) #递归调用
            else:  classLabel=secondDict[key]#为叶子结点，赋予label值
    return classLabel #分类结果

#决策树的存储
def storeTree(inputTree,filename):#序列化的对象可以在磁盘上保存，需要时读取
    import pickle #python序列化对象，这里序列化保存树结构的字典对象
    fw=open(filename,'wb') #wirte()error 讲书上改成‘wb’
    pickle.dump(inputTree,fw)
    fw.close()

#与决策树的存储对应，读取决策树
def grabTree(filename): #读取对象
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)


#下面的代码为本人自己测试决策树功能的代码，实际调用时需要删除
if __name__ == "__main__":
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    #print(myTree)
    storeTree(myTree, 'aa.txt')
    b = grabTree('aa.txt')
    print(classify(myTree,labels,['青年','否','否','一般']))
















