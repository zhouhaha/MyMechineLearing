function x=init_individual(xlb,xub,dim,sizepop)
% 参数初始化函数
% lb：参数下界，行向量
% ub：参数上界，行向量
% dim：参数维度
% sizepop 种群规模
% x：返回sizepop*size(lb,2)的参数矩阵
xRange=repmat((xub-xlb),[sizepop,1]);
xLower=repmat(xlb,[sizepop,1]);
x=rand(sizepop,dim).*xRange+xLower;