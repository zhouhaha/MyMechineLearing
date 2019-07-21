#coding=utf-8
from numpy import *
#文本转化为词向量
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1表示侮辱类，0表示不属于
    return postingList,classVec #词条切分后的分档和类别标签

#返回一个list，该list包含输入的数据（以set（）类型转换成不含重复类型）的所有文档矩阵
def createVocabList(dataSet):
    vocabSet=set([])#创建空集，set是返回不带重复词的list
    for document in dataSet:
        vocabSet=vocabSet|set(document) #创建两个集合的并集
    return list(vocabSet)
#判断某个词条在文档中是否出现，如果出现，则vocabList中某个文字对应的位置置为1
def setOfWords2Vec(vocabList, inputSet):#输入参数为词汇表和某个文档，inputset为一列，而不是一个矩阵
    returnVec = [0]*len(vocabList)    #初始化returnVec，其为一个长度与vocabList相同的数组
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1   #返回检索的索引值
        else: print("the word: %s is not in my Vocabulary!" % word)#返回文档向量 表示某个词是否在输入文档中出现过 1/0
    return returnVec
#高级词袋模型，判断词出现次数
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)  # 返回文档向量 表示某个词是否在输入文档中出现过 1/0
    return returnVec

#朴素贝叶斯分类训练函数，返回三个数，一个为标签为1的对应概率，
# 一个为标签为1的情况下单词的比例构成的数组，一个为标签为0的情况下单词所占比例构成的数组
def trainNB0(trainMatrix,trainCategory):  #输入参数为文档矩阵以及其对应的类别标签，trainMatrix存储的是文档对应的每个字符所出现的次数构成的矩阵
    numTrainDocs=len(trainMatrix) #文档行数
    numWords=len(trainMatrix[0])  #每一行文档的文字数量
    pAbusive=sum(trainCategory)/float(numTrainDocs) #文档中属于侮辱类的概率，侮辱类为1，0是非侮辱类，总和除以总个数
    #p0Num=zeros(numWords); p1Num=zeros(numWords)
    #p0Denom=0.0;p1Denom=0.0
    p0Num = zeros(numWords)  #定义一个长度与文档列数相同的数组
    p1Num = zeros(numWords)  #定义一个长度与文档列数相同的数组
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):#遍历每篇文档
        #if else潜在遍历类别，共2个类别
        if trainCategory[i]==1: #一旦某个词出现在某个文档中出现（出现为1，不出现为0）
            p1Num+=trainMatrix[i]  #该词数加1  这是数组之间的对应的元素相加（仅限于numpy库中的array([])类型），可以求得整篇文档中的某个单词的数量
            p1Denom+=sum(trainMatrix[i]) #文档总词数
        else: #另一个类别
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
        # p1Vect = p1Num / p1Denom
        # p0Vect = p0Num / p0Denom
    # 表示在一组样本文档中（每个文档是一行），对于标签为1或者0的情况下，统计某个单词在文档中是否出现的
    #总个数，然后除以总类别，获得在已知样本标签的情况下某个单词是否出现的概率
    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive  #返回p0Vec，p1Vec都是矩阵，对应每个词在文档总体中出现概率，pAb对应文档属于1的概率

#给定词向量 判断类别，输入参数第一个所需判断类别的文档对应字段，后面三个为训练函数返回的三个
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1): #第一个参数为0,1组合二分类矩阵，对应词汇表各个词是否出现
    # 对应元素相乘，然后求和。得到的是在给定参数（w1，w2。。。）返回该文档属于某一类的概率，注意这里是对数相加，对应的为概率的乘积
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else: return 0
#封装的bayes测试函数
def testingNB():
    listOPosts,listClasses=loadDataSet() #导入数据，第一个存储文档，第二个存储文档标记类别
    myVocabList=createVocabList(listOPosts) #所有词汇总list，不含重复的，类似于建立词汇表
    trainMat=[]
    for postinDoc in listOPosts:#生成文档对应词的矩阵 每个文档一行，每行内容为词向量
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc)) #每个词在文档中是否出现，生成1、0组合的词向量
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses)) #根据现有数据输出词对应的类别判定和概率
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry)) #判断测试词条在词汇list中是否出现，生成词向量
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)) #根据贝叶斯返回的概率，将测试向量与之乘，输出结果
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

#示例：过滤垃圾邮件
#预处理，将传入的字符串bigString拆分成单个字母，并保存在一个list中，且只保留了长度大于2的字符
def textParse(bigString):
    import re
    listOfTokens=re.split('x*',bigString)  #接收一个大字符串并将其解析为字符串列表,返回的是一个list类型
    return [tok.lower() for tok in listOfTokens if len(tok)>2] #去掉少于两个的字符串并全部转化为小写
#过滤邮件 训练+测试
def spamTest():
    docList=[]; classList=[]; fullText=[]
    #for循环是将文件读取出来存储到上面三个list中
    for i in range(1,26):
        #wordList=textParse(open('email/spam/%d.txt' %i).read()) 书上这行代码有些问题 unicode error
        #修改为下面：
        #spam为标签为1
        wordList = textParse(open('email/spam/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))  #设置编码格式，ignore为报错时的解决方案，注意对比read/readline/readlines
        #read返回整个文档读取的字符，readline返回读取的单列字符；readlines按列返回，最后形成一个list，保存每一行的字符
        docList.append(wordList)  #append保留添加的原格式，即，有可能变为二维数组,如[1,2].append([3,4])变为[1,2,[3,4]]
        fullText.extend(wordList)  #extend不保留原格式，直接加在后面，如[1,2].extend([3,4])变为[1,2,3,4]
        classList.append(1)
        # wordList=textParse(open('email/ham/%d.txt' %i).read()) 同理上面一样 修改为下面一行
        #ham标签为0
        wordList = textParse(open('email/ham/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList) #不融合格式
        fullText.extend(wordList) #添加元素 去掉数组格式
        classList.append(0)
    vocabList=createVocabList(docList) #创建词列表
    trainingSet = list(range(50))  #建立一个0,1,2.。。。49的长度为50的数组
    #trainingSet=range(50) python3 del不支持返回数组对象 而是range对象
    testSet=[] #spam+ham=50 eamils
    for i in range(10):#随机选择10封作为测试集
        randIndex=int(random.uniform(0,len(trainingSet)))  #从0-50中随机选择一个数
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) #报错，python3 del不支持返回数组对象 而是range对象 修改上面108行   抽取一封信之后，删除原有的位置
    trainMat=[]; trainClasses=[]
    for docIndex in trainingSet: #遍历训练集
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex])) #对每一封邮件创建词向量并计算分类概率
        trainClasses.append(classList[docIndex]) #类别
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses)) #训练出概率
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is:',float(errorCount)/len(testSet))

#从个人广告中获取区域倾向
#RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):#对所有词出现频率进行排序，返回排序后出现频率最高的前30个
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)#True=降序排列
    return sortedFreq[:30]

def localWords(feed1,feed0):#两个RSS源作为参数,与spamTest差别不大
    import feedparser
    docList=[];classList=[];fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):#访问RSS源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:#去掉出现频数最高的钱30个词
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen)) #python3修改替换trainSet=range(2*minLen)
    testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

#最具代表性的词汇显示函数
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-1.0: topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-1.0: topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])


























