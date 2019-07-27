#coding=utf-8
from numpy import *
from time import sleep
import math
import scipy.io as sio

#加载数据和数据标签，输入参数是一个txt的文件，返回的是一个list类型的dataMat存储特征；一个list类型的labelMat存储特征对应的状态标签
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#输入参数为alpha的一个下标i和总数m，返回的是随机得到的下标j
def selectJrand(i,m): #i表示alpha的下标，m表示alpha的总数
    j=i
    while(j==i):
        j=int(random.uniform(0,m)) #简化版SMO，alpha随机选择
    return j

#辅助函数，用于调整大于H或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

#简单的SMO算法，函数的五个输入分别为：数据集、类别标签、常数C、容错率和退出前最大的循环次数，返回的是一个matrix类型的矩阵alphs和常量b
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):#toler表示容错率 常数C
    dataMatrix = mat(dataMatIn)  #将特征集合变为矩阵
    labelMat = mat(classLabels).transpose()  #将对应的标签集合也变成矩阵
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))  #alpha是一个长度为m，即等于样本个数的矩阵
    iter = 0   #表示当前迭代次数
    while (iter < maxIter):
        alphaPairsChanged = 0   #  标记alpha是否被优化
        for i in range(m):
            # fXi是预测的类别
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  #np中multiply表示对应元素相乘，该条语句对应的是fxi = wx+b，所有fxi是一个值
            # Ei表示误差
            Ei = fXi - float(labelMat[i])# 预测结果和真实结果比对，计算误差
            # 对alpha进行优化，同时检查alpha的值满足两个条件：if
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)# 随机选择第二个alpha
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy(); # 分配内存 稍后比较误差
                if (labelMat[i] != labelMat[j]): # 计算L H用于将alpha[j]调整到0—C之间
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print ("L==H"); continue  # eta为alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print ("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 检查alpha[j]是否有轻微改变，如果改变很小，即可认为alphas[i]不需要改变了
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print ("j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas

#一个核函数，将低维度数据映射至高维度数据
#输入参数X为特征数据集，为一个m*n的矩阵，A表示一个1*n的矩阵，kTup表示一个元组
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)     #X is the type of kernel and the other two parameters are optional parameter
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel  #线性核函数
    elif kTup[0]=='rbf':  #径向基核函数
        for j in range(m): #compute the guassian
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #numpy中除法意味着对矩阵展开计算而不是matlab的求矩阵的逆
    #如果遇到无法识别的元祖，程序抛出异常
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:  #保存所有重要值，实现对成员变量的填充
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # 第四个参数为容错率，第五个参数为
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]  #获得输入的样本行数
        self.alphas = mat(zeros((self.m,1)))   #定义alpha的长度与样本个数相同
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #误差缓存
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

#计算误差,oS是指optStruct这个类的对象
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei): #选择第二个alpha值以保证每次优化的最大步长（内循环）change English~
    maxK=-1
    maxDeltaE=0
    Ej=0
    oS.eCache[i]=[1,Ei] #input Ei
    validEcacheList=nonzero(oS.eCache[:,0].A)[0] #set Ei valid in cache，nonzero is a list
    #choose the num which change most, if the first choice: use random
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k; maxDeltaE=deltaE; Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print ("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

#full Platt SMO
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    # 当迭代超过maxIter或者整个数据集都未对任意alpha进行修改时，退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)#use innerL to choose the second alpha
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

#根据得到的alpha得到权值w，计算公式为：w=sum(alpha[i]*label[i]*X[i]),返回的是一个与X长度相同的m*1的矩阵
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def testRbf(k1=0.1):
    dataArr,labelArr=loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important  返回的是一个矩阵形式的
    datMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0] #找到非零的alpha值，从而得到所需的支持向量，返回支持向量对应的检索值
    #支持向量：数据点指向超平面的长度表示距离的方向向量
    sVs=datMat[svInd]
    labelSV = labelMat[svInd] #alpha类别标签值
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b   #对应wx+b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    #不同数据集
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


#示例：手写识别
#准备数据：把二值化图像转化为向量
#按照每一行读取txt文件中的32*32像素所代表的0/1值，并将其转为一个1*1024的数组里面，返回所存储的1*1024的数组

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
#收集数据：导入数据集
def loadImages(dirName):
    # from os import listdir
    # import sys
    import os,sys
    hwLabels=[]
    trainingFileList=os.listdir(dirName)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        #这里是二分类问题，只分类数字1和9，数字分类结果为9时返回-1
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels
#测试算法（调用训练算法函数smoP）
if __name__ =='__main__':
    kTup = ('rbf', 1)
    dataArr, labelArr = loadImages('trainingDigits')
    print(dataArr, labelArr)
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


def testDigits(kTup=('rbf', 10)): #与之前的testrbf函数差别不大，loadImages和核函数kTup选择输入，默认rbf类别
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd]
    print ("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))

    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m) )











