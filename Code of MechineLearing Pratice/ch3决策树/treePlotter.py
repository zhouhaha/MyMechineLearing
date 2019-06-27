#coding=utf-8
#改文件功能为对决策树进行绘制，图形化
import operator
from ch3决策树 import tree
import matplotlib.pyplot as plt
#绘制属性图
plt.rcParams['font.family'] = 'STSong'
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  #决策节点的属性。boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
leafNode = dict(boxstyle="round4", fc="0.8") #决策树叶子节点的属性
arrow_args = dict(arrowstyle="<-") #箭头的属性

#构造注解树 在python字典形式中如何存储树
#获取叶节点数目
def getNumLeafs(myTree):
    numLeafs=0 #初始化结点数
    # 下面三行为代码 python3 替换注释的两行代码
    firstSides = list(myTree.keys())  #将决策树的键值对中的键选取出来保存至firstSides中
    firstStr = firstSides[0]  # 找到输入的第一个元素,第一个关键词为划分数据集类别的标签
    secondDict = myTree[firstStr] #第一个键对应的值，也有可能是一个dict
    #firstStr = list(myTree)
    #secondDict=myTree[firstStr]
    for key in secondDict.keys(): #测试数据是否为字典形式
        if type(secondDict[key]).__name__=='dict': #type判断子结点是否为字典类型
            numLeafs+=getNumLeafs(secondDict[key])
            #若子节点也为字典，则也是判断结点，需要递归获取num
        else:  numLeafs+=1
    return numLeafs #返回整棵树的叶结点数
#获取树的层数
def getTreeDepth(myTree):
    maxDepth=0
    # 下面三行为代码 python3 替换注释的两行代码
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    #firstStr=myTree.keys()[0]
    #secondDict=myTree[firstStr]#获取划分类别的标签
    for key in secondDict.keys():
        if type(secondDict[key]).__name__== 'dict':
           thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

#绘制节点，其中nodeTxt为绘制的文本内容，centerPt为箭头所指的地方坐标，parentPt为箭头非箭头端的坐标，nodeType为节点的绘制属性
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
# nodeTxt为要显示的文本，centerPt为文本的中心点，parentPt为箭头指向文本的点，xy是箭头尖的坐标，xytest设置注释内容显示的中心位置
# xycoords和textcoords是坐标xy与xytext的说明（按轴坐标），若textcoords=None，则默认textcoords与xycoords相同，若都未设置，默认为data
# va/ha设置节点框中文字的位置，va为纵向取值为(u'top', u'bottom', u'center', u'baseline')，ha为横向取值为(u'center', u'right', u'left')

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

#画树，被creatPlot（）函数引用
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  #计算树的宽度  totalW
    depth = getTreeDepth(myTree) #计算树的高度 存储在totalD
    #python3.x修改
    firstSides = list(myTree.keys())#firstStr = myTree.keys()[0]     #the text label for this node should be this
    firstStr = firstSides[0]  # 找到输入的第一个元素
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)#按照叶子结点个数划分x轴
    plotMidText(cntrPt, parentPt, nodeTxt) #标注结点属性
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #y方向上的摆放位置 自上而下绘制，因此递减y值
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#判断是否为字典 不是则为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))        #递归继续向下找
        else:   #为叶子结点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW #x方向计算结点坐标
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)#绘制
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))#添加文本信息
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD #下次重新调用时恢复y

#画出决策树
def createPlot(inTree): #主函数
    fig = plt.figure(1, facecolor='white')  #创建一个一个画布，背景为白色
    fig.clf()  #清空画布
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) #ax1是函数createPlot的一个属性，这个可以在函数里面定义也可以在函数定义后加入也可以
    # frameon表示是否绘制坐标轴矩形
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

#输出预先存储的树信息，避免每次测试都需要重新创建树
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: {"pigu":{"zhou":'one','zhazh':'two','gua':'san'}}, 1: {'flippers': {0: {'head': {0: 'no', 1:'yes',2:'yoxi'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


#下面的代码为本人自己测试决策树功能的代码，实际调用时需要删除
if __name__  =="__main__":
    dataSet, labels = tree.createDataSet()
    myTree = tree.createTree(dataSet, labels)
    #myTree = retrieveTree(1)
    a = getNumLeafs(myTree)
    b = getTreeDepth(myTree)
    c = createPlot(myTree)











