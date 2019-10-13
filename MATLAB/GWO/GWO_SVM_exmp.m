tic % ��ʱ��
%% ��ջ�������
close all
clear
clc
format compact
%% ������ȡ
% �����������wine,���а���������Ϊclassnumber = 3,wine:178*13�ľ���,wine_labes:178*1��������
load wine.mat
% ѡ��ѵ�����Ͳ��Լ�
% ����һ���1-30,�ڶ����60-95,�������131-153��Ϊѵ����
train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% ��Ӧ��ѵ�����ı�ǩҲҪ�������
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% ����һ���31-59,�ڶ����96-130,�������154-178��Ϊ���Լ�
test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% ��Ӧ�Ĳ��Լ��ı�ǩҲҪ�������
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];
%% ����Ԥ����
% ����Ԥ����,��ѵ�����Ͳ��Լ���һ����[0,1]����
[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);

dataset = [train_wine;test_wine];
% mapminmaxΪMATLAB�Դ��Ĺ�һ������
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% ���û����㷨ѡ����ѵ�SVM����c��g
SearchAgents_no=10; % ��Ⱥ������Number of search agents
Max_iteration=10; % ������������Maximum numbef of iterations
dim=2; % ������Ҫ�Ż���������c��g��number of your variables
lb=[0.01,0.01]; % ����ȡֵ�½�
ub=[100,100]; % ����ȡֵ�Ͻ�
% v = 5; % SVM Cross Validation����,Ĭ��Ϊ5

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim); % ��ʼ��Alpha�ǵ�λ��
Alpha_score=inf; % ��ʼ��Alpha�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems

Beta_pos=zeros(1,dim); % ��ʼ��Beta�ǵ�λ��
Beta_score=inf; % ��ʼ��Beta�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems

Delta_pos=zeros(1,dim); % ��ʼ��Delta�ǵ�λ��
Delta_score=inf; % ��ʼ��Delta�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iteration);

l=0; % Loop counterѭ��������

% Main loop��ѭ��
while l<Max_iteration  % �Ե�������ѭ��
    for i=1:size(Positions,1)  % ����ÿ����
        
       % Return back the search agents that go beyond the boundaries of the search space
       % ������λ�ó����������ռ䣬��Ҫ���»ص������ռ�
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        % ���ǵ�λ�������ֵ����Сֵ֮�䣬��λ�ò���Ҫ���������������ֵ����ص����ֵ�߽磻
        % ��������Сֵ����ش���Сֵ�߽�
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % ~��ʾȡ��           
     
      % ������Ӧ�Ⱥ���ֵ
       cmd = [' -c ',num2str(Positions(i,1)),' -g ',num2str(Positions(i,2))];
       model=svmtrain(train_wine_labels,train_wine,cmd); % SVMģ��ѵ��
       %[~,fitness]=svmpredict(test_wine_labels,test_wine,model); % SVMģ��Ԥ�⼰�侫��
       [~,~,fitness]=svmpredict(test_wine_labels,test_wine,model);% SVMģ��Ԥ�⼰�侫��
       fitness=100-fitness(1); % �Դ�������С��ΪĿ��
    
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score % ���Ŀ�꺯��ֵС��Alpha�ǵ�Ŀ�꺯��ֵ
            Alpha_score=fitness; % ��Alpha�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ��Update alpha
            Alpha_pos=Positions(i,:); % ͬʱ��Alpha�ǵ�λ�ø���Ϊ����λ��
        end
        
        if fitness>Alpha_score && fitness<Beta_score % ���Ŀ�꺯��ֵ������Alpha�Ǻ�Beta�ǵ�Ŀ�꺯��ֵ֮��
            Beta_score=fitness; % ��Beta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ��Update beta
            Beta_pos=Positions(i,:); % ͬʱ����Beta�ǵ�λ��
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score  % ���Ŀ�꺯��ֵ������Beta�Ǻ�Delta�ǵ�Ŀ�꺯��ֵ֮��
            Delta_score=fitness; % ��Delta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ��Update delta
            Delta_pos=Positions(i,:); % ͬʱ����Delta�ǵ�λ��
        end
    end
    
    a=2-l*((2)/Max_iteration); % ��ÿһ�ε�����������Ӧ��aֵ��a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1) % ����ÿ����
        for j=1:size(Positions,2) % ����ÿ��ά��
            
            % ��Χ���λ�ø���
            
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % ����ϵ��A��Equation (3.3)
            C1=2*r2; % ����ϵ��C��Equation (3.4)
            
            % Alpha��λ�ø���
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % ����ϵ��A��Equation (3.3)
            C2=2*r2; % ����ϵ��C��Equation (3.4)
            
            % Beta��λ�ø���
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % ����ϵ��A��Equation (3.3)
            C3=2*r2; % ����ϵ��C��Equation (3.4)
            
            % Delta��λ�ø���
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            % λ�ø���
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    l=l+1;    
    Convergence_curve(l)=Alpha_score;
end
bestc=Alpha_pos(1,1);
bestg=Alpha_pos(1,2);
bestGWOaccuarcy=Alpha_score;
%% ��ӡ����ѡ����
disp('��ӡѡ����');
str=sprintf('Best Cross Validation Accuracy = %g%%��Best c = %g��Best g = %g',bestGWOaccuarcy*100,bestc,bestg);
disp(str)
%% ������ѵĲ�������SVM����ѵ��
cmd_gwosvm = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model_gwosvm = svmtrain(train_wine_labels,train_wine,cmd_gwosvm);
%% SVM����Ԥ��
[predict_label,accuracy] = svmpredict(test_wine_labels,test_wine,model_gwosvm);
% ��ӡ���Լ�����׼ȷ��
total = length(test_wine_labels);
right = sum(predict_label == test_wine_labels);
disp('��ӡ���Լ�����׼ȷ��');
str = sprintf( 'Accuracy = %g%% (%d/%d)',accuracy(1),right,total);
disp(str);
%% �������
% ���Լ���ʵ�ʷ����Ԥ�����ͼ
figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_label,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on
snapnow
%% ��ʾ��������ʱ��
toc