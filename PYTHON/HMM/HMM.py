import numpy as np
import csv

class HMM(object):
    def __init__(self,N,M):
        self.A = np.zeros((N,N))        # 状态转移概率矩阵，初始化为N*N的0矩阵
        self.B = np.zeros((N,M))        # 观测概率矩阵，初始化为N*M的0矩阵

        self.N = N                      # 可能的状态数,隐藏状态个数
        self.M = M                      # 可能的观测数，显示状态个数

    def cal_probality(self, O):
        self.T = len(O)
        self.O = O                      #O为观测的状态，已知

        self.forward()
        return sum(self.alpha[self.T-1])

    #描述HMM的前向算法公式代码实现：获得一个二维矩阵alpha表示，横表示时间，列表示状态，参看前向算法的定义，在A、B固定的情况下
    def forward(self):
        """
        前向算法
        """
        self.alpha = np.zeros((self.T,self.N))         #self.alpha是一个二维数组，其中第一个[]表示所述的时间t时刻，第二个[]表示t时刻第i个状态

        # 公式 10.15求α1
        for i in range(self.N):
            self.alpha[0][i] = self.Pi[i]*self.B[i][self.O[0]]    #描述第一个α的值

        # 公式10.16
        for t in range(1,self.T):
            for i in range(self.N):
                sum = 0
                for j in range(self.N):
                    sum += self.alpha[t-1][j]*self.A[j][i]
                self.alpha[t][i] = sum * self.B[i][self.O[t]]

    #描述HMM的后向算法公式代码实现：获得一个二维矩阵beta表示，横表示时间，列表示状态，参看后向算法的定义，考虑A、B固定
    def backward(self):
        """
        后向算法
        """
        self.beta = np.zeros((self.T,self.N))

        # 公式10.19
        for i in range(self.N):
            self.beta[self.T-1][i] = 1

        # 公式10.20
        for t in range(self.T-2,-1,-1):
            for i in range(self.N):
                for j in range(self.N):
                    self.beta[t][i] += self.A[i][j]*self.B[j][self.O[t+1]]*self.beta[t+1][j]

    #返回t时刻隐藏状态为i的概率
    def cal_gamma(self, i, t):
        """
        公式 10.24  # 该函数返回在t时刻隐藏状态为i的概率
        """
        numerator = self.alpha[t][i]*self.beta[t][i]
        denominator = 0

        for j in range(self.N):
            denominator += self.alpha[t][j]*self.beta[t][j]

        return numerator/denominator

    #返回在t时刻隐藏状态为i，t+1时刻隐藏状态为j的概率
    def cal_ksi(self, i, j, t):
        """
        公式 10.26  返回在t时刻隐藏状态为i，t+1时刻隐藏状态为j的概率
        """

        numerator = self.alpha[t][i]*self.A[i][j]*self.B[j][self.O[t+1]]*self.beta[t+1][j]
        denominator = 0

        for i in range(self.N):
            for j in range(self.N):
                denominator += self.alpha[t][i]*self.A[i][j]*self.B[j][self.O[t+1]]*self.beta[t+1][j]

        return numerator/denominator

    # 初始化状态转移概率矩阵A和状态观测概率矩阵B，初始状态概率π，并保证每行相加等于1
    def init(self):
        """
        随机生成 A，B，Pi
        并保证每行相加等于 1
        """
        import random
        self.Pi = np.array([1.0 / self.N] * self.N)  # 初始状态概率矩阵
        #初始化状态转移矩阵A，随机取0~100之间的数给A[N][N],并且每一行相加之和为1
        for i in range(self.N):
            randomlist = [random.randint(0,100) for t in range(self.N)]   #randomlist返回一个长度为N的数组
            Sum = sum(randomlist)    #对数组randomlist求和
            for j in range(self.N):
                self.A[i][j] = randomlist[j]/Sum

        # 初始化状态观测矩阵B，随机取0~100之间的数给B[N][M],并且每一行相加之和为1
        for i in range(self.N):
            randomlist = [random.randint(0,100) for t in range(self.M)]
            Sum = sum(randomlist)
            for j in range(self.M):
                self.B[i][j] = randomlist[j]/Sum

    # 利用Baum-Welch算法对隐马尔科夫模型的A、B以及π进行计算和训练，训练次数为MaxSteps
    def train(self, O, MaxSteps = 100):
        self.T = len(O)
        self.O = O

        # 初始化A、B和pi
        self.init()

        step = 0
        # 递推
        while step<MaxSteps:  #最大训练次数为100次
            step+=1
            print(step)
            tmp_A = np.zeros((self.N,self.N))
            tmp_B = np.zeros((self.N,self.M))
            tmp_pi = np.array([0.0]*self.N)

            self.forward()  #进行一次计算，更新一次前向算法的二维矩阵
            self.backward()  #进行一次计算，更新一次后向算法的二维矩阵

            # 训练一次状态转移矩阵a_{ij}
            for i in range(self.N):
                for j in range(self.N):
                    numerator=0.0
                    denominator=0.0
                    for t in range(self.T-1):
                        numerator += self.cal_ksi(i,j,t)#隐藏状态为i，t+1时刻隐藏状态为j的概率之和
                        denominator += self.cal_gamma(i,t)  #隐藏状态为i的概率之和
                    tmp_A[i][j] = numerator/denominator

            # 训练一次状态观测矩阵b_{jk}
            for j in range(self.N):
                for k in range(self.M):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(self.T):
                        if k == self.O[t]:
                            numerator += self.cal_gamma(j,t)
                        denominator += self.cal_gamma(j,t)
                    tmp_B[j][k] = numerator / denominator

            # pi_i
            for i in range(self.N):
                tmp_pi[i] = self.cal_gamma(i,0)

            self.A = tmp_A
            self.B = tmp_B
            self.Pi = tmp_pi

    def generate(self, length):
        import random
        I = []
        # start
        ran = random.randint(0,1000)/1000.0  #产生一个范围在（0~1）之间的随机数
        i = 0
        while self.Pi[i]<ran or self.Pi[i]<0.0002:  #直到所有的pi[i]都大于或等于ran或者0.0001时，才停止
            ran -= self.Pi[i]
            i += 1

        I.append(i)
        # 生成状态序列
        for k in range(1,length):
            last = I[-1]
            ran = random.randint(0, 1000) / 1000.0
            i = 0
            while self.A[last][i] < ran or self.A[last][i]<0.0002:
                ran -= self.A[last][i]
                i += 1
            I.append(i)
        # 生成观测序列
        Y = []
        for i in range(length):
            k = 0
            ran = random.randint(0, 1000) / 1000.0
            while self.B[I[i]][k] < ran or self.B[I[i]][k]<0.0002:
                ran -= self.B[I[i]][k]
                k += 1
            Y.append(k)

        return Y

#生成一个三角波，用于做预测的数据
def triangle(length):
    '''
    定义一个三角波，返回X为三角波的横轴，Y值为三角波的纵轴
    '''
    X = [i for i in range(length)]
    Y = []

    for x in X:
        x = x % 6
        if x <= 3:
            Y.append(x)
        else:
            Y.append(6-x)
    return X,Y


#将数据通过matplotlib库显示
def show_data(x,y):
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'r')
    plt.show()

    return y

if __name__ == '__main__':
    hmm = HMM(10,4)
    tri_x, tri_y = triangle(20)

    hmm.train(tri_y)
    y = hmm.generate(100)
    print(y)
    x = [i for i in range(100)]
    show_data(x,y)
