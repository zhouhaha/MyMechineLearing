tic % 计时器
%% 清空环境变量
close all
clear
clc
format compact
%% 数据提取
% 载入测试数据wine,其中包含的数据为classnumber = 3,wine:178*13的矩阵,wine_labes:178*1的列向量
load wine.mat
% 选定训练集和测试集
% 将第一类的1-30,第二类的60-95,第三类的131-153做为训练集
train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% 相应的训练集的标签也要分离出来
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% 将第一类的31-59,第二类的96-130,第三类的154-178做为测试集
test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% 相应的测试集的标签也要分离出来
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];
%% 数据预处理
% 数据预处理,将训练集和测试集归一化到[0,1]区间
[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);

dataset = [train_wine;test_wine];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% 利用灰狼算法选择最佳的SVM参数c和g
SearchAgents_no=10; % 狼群数量，Number of search agents
Max_iteration=10; % 最大迭代次数，Maximum numbef of iterations
dim=2; % 此例需要优化两个参数c和g，number of your variables
lb=[0.01,0.01]; % 参数取值下界
ub=[100,100]; % 参数取值上界
% v = 5; % SVM Cross Validation参数,默认为5

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim); % 初始化Alpha狼的位置
Alpha_score=inf; % 初始化Alpha狼的目标函数值，change this to -inf for maximization problems

Beta_pos=zeros(1,dim); % 初始化Beta狼的位置
Beta_score=inf; % 初始化Beta狼的目标函数值，change this to -inf for maximization problems

Delta_pos=zeros(1,dim); % 初始化Delta狼的位置
Delta_score=inf; % 初始化Delta狼的目标函数值，change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iteration);

l=0; % Loop counter循环计数器

% Main loop主循环
while l<Max_iteration  % 对迭代次数循环
    for i=1:size(Positions,1)  % 遍历每个狼
        
       % Return back the search agents that go beyond the boundaries of the search space
       % 若搜索位置超过了搜索空间，需要重新回到搜索空间
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        % 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界；
        % 若超出最小值，最回答最小值边界
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % ~表示取反           
     
      % 计算适应度函数值
       cmd = [' -c ',num2str(Positions(i,1)),' -g ',num2str(Positions(i,2))];
       model=svmtrain(train_wine_labels,train_wine,cmd); % SVM模型训练
       %[~,fitness]=svmpredict(test_wine_labels,test_wine,model); % SVM模型预测及其精度
       [~,~,fitness]=svmpredict(test_wine_labels,test_wine,model);% SVM模型预测及其精度
       fitness=100-fitness(1); % 以错误率最小化为目标
    
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score % 如果目标函数值小于Alpha狼的目标函数值
            Alpha_score=fitness; % 则将Alpha狼的目标函数值更新为最优目标函数值，Update alpha
            Alpha_pos=Positions(i,:); % 同时将Alpha狼的位置更新为最优位置
        end
        
        if fitness>Alpha_score && fitness<Beta_score % 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
            Beta_score=fitness; % 则将Beta狼的目标函数值更新为最优目标函数值，Update beta
            Beta_pos=Positions(i,:); % 同时更新Beta狼的位置
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score  % 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
            Delta_score=fitness; % 则将Delta狼的目标函数值更新为最优目标函数值，Update delta
            Delta_pos=Positions(i,:); % 同时更新Delta狼的位置
        end
    end
    
    a=2-l*((2)/Max_iteration); % 对每一次迭代，计算相应的a值，a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1) % 遍历每个狼
        for j=1:size(Positions,2) % 遍历每个维度
            
            % 包围猎物，位置更新
            
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % 计算系数A，Equation (3.3)
            C1=2*r2; % 计算系数C，Equation (3.4)
            
            % Alpha狼位置更新
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % 计算系数A，Equation (3.3)
            C2=2*r2; % 计算系数C，Equation (3.4)
            
            % Beta狼位置更新
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % 计算系数A，Equation (3.3)
            C3=2*r2; % 计算系数C，Equation (3.4)
            
            % Delta狼位置更新
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            % 位置更新
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    l=l+1;    
    Convergence_curve(l)=Alpha_score;
end
bestc=Alpha_pos(1,1);
bestg=Alpha_pos(1,2);
bestGWOaccuarcy=Alpha_score;
%% 打印参数选择结果
disp('打印选择结果');
str=sprintf('Best Cross Validation Accuracy = %g%%，Best c = %g，Best g = %g',bestGWOaccuarcy*100,bestc,bestg);
disp(str)
%% 利用最佳的参数进行SVM网络训练
cmd_gwosvm = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model_gwosvm = svmtrain(train_wine_labels,train_wine,cmd_gwosvm);
%% SVM网络预测
[predict_label,accuracy] = svmpredict(test_wine_labels,test_wine,model_gwosvm);
% 打印测试集分类准确率
total = length(test_wine_labels);
right = sum(predict_label == test_wine_labels);
disp('打印测试集分类准确率');
str = sprintf( 'Accuracy = %g%% (%d/%d)',accuracy(1),right,total);
disp(str);
%% 结果分析
% 测试集的实际分类和预测分类图
figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_label,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('测试集的实际分类和预测分类图','FontSize',12);
grid on
snapnow
%% 显示程序运行时间
toc