function x=init_individual(xlb,xub,dim,sizepop)
% ������ʼ������
% lb�������½磬������
% ub�������Ͻ磬������
% dim������ά��
% sizepop ��Ⱥ��ģ
% x������sizepop*size(lb,2)�Ĳ�������
xRange=repmat((xub-xlb),[sizepop,1]);
xLower=repmat(xlb,[sizepop,1]);
x=rand(sizepop,dim).*xRange+xLower;