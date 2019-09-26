function [bestc,bestg,test_pre]=my_HGWO_SVR(para,input_train,output_train,input_test,output_test)
% �������� parameters [n,N_iteration,beta_min,beta_max,pCR]
% nΪ��Ⱥ��ģ��N_iterationΪ��������
% beta_min ���������½� Lower Bound of Scaling Factor
% beta_max=0.8; % ���������Ͻ� Upper Bound of Scaling Factor
% pCR ������� Crossover Probability
% Ҫ����������Ϊ������������

%% ���ݹ�һ��
[input_train,rule1]=mapminmax(input_train');
[output_train,rule2]=mapminmax(output_train');
input_test=mapminmax('apply',input_test',rule1);
output_test=mapminmax('apply',output_test',rule2);
input_train=input_train';
output_train=output_train';
input_test=input_test';
output_test=output_test';
%% ���ò�ֽ���-�����Ż�����㷨��DE_GWO��ѡ����ѵ�SVR����
nPop=para(1); % ��Ⱥ��ģ Population Size
MaxIt=para(2); % ����������Maximum Number of Iterations
nVar=2; % �Ա���ά����������Ҫ�Ż���������c��g Number of Decision Variables
VarSize=[1,nVar]; % ���߱��������С Decision Variables Matrix Size
beta_min=para(3); % ���������½� Lower Bound of Scaling Factor
beta_max=para(4); % ���������Ͻ� Upper Bound of Scaling Factor
pCR=para(5); %  ������� Crossover Probability
lb=[0.01,0.01]; % ����ȡֵ�½�
ub=[100,100]; % ����ȡֵ�Ͻ�
%% ��ʼ��
% ������Ⱥ��ʼ��
parent_Position=init_individual(lb,ub,nVar,nPop); % �����ʼ��λ��
parent_Val=zeros(nPop,1); % Ŀ�꺯��ֵ
for i=1:nPop % ����ÿ������
    parent_Val(i)=fobj(parent_Position(i,:),input_train,output_train,input_test,output_test); % �������Ŀ�꺯��ֵ
end
% ͻ����Ⱥ��ʼ��
mutant_Position=init_individual(lb,ub,nVar,nPop); % �����ʼ��λ��
mutant_Val=zeros(nPop,1); % Ŀ�꺯��ֵ
for i=1:nPop % ����ÿ������
    mutant_Val(i)=fobj(mutant_Position(i,:),input_train,output_train,input_test,output_test); % �������Ŀ�꺯��ֵ
end
% �Ӵ���Ⱥ��ʼ��
child_Position=init_individual(lb,ub,nVar,nPop); % �����ʼ��λ��
child_Val=zeros(nPop,1); % Ŀ�꺯��ֵ
for i=1:nPop % ����ÿ������
    child_Val(i)=fobj(child_Position(i,:),input_train,output_train,input_test,output_test); % �������Ŀ�꺯��ֵ
end
%% ȷ��������Ⱥ�е�Alpha,Beta,Delta��
[~,sort_index]=sort(parent_Val); % ������ȺĿ�꺯��ֵ����
parent_Alpha_Position=parent_Position(sort_index(1),:); % ȷ������Alpha��
parent_Alpha_Val=parent_Val(sort_index(1)); % ����Alpha��Ŀ�꺯��ֵ
parent_Beta_Position=parent_Position(sort_index(2),:); % ȷ������Beta��
parent_Delta_Position=parent_Position(sort_index(3),:); % ȷ������Delta��
%% ������ʼ
BestCost=zeros(1,MaxIt);
BestCost(1)=parent_Alpha_Val;
for it=1:MaxIt
    a=2-it*((2)/MaxIt); % ��ÿһ�ε�����������Ӧ��aֵ��a decreases linearly fron 2 to 0
    % ���¸�������λ��
    for par=1:nPop % ������������
        for var=1:nVar % ����ÿ��ά��            
            % Alpha��Hunting
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]            
            A1=2*a*r1-a; % ����ϵ��A
            C1=2*r2; % ����ϵ��C
            D_alpha=abs(C1*parent_Alpha_Position(var)-parent_Position(par,var));
            X1=parent_Alpha_Position(var)-A1*D_alpha;
            % Beta��Hunting
            r1=rand();
            r2=rand();            
            A2=2*a*r1-a; % ����ϵ��A
            C2=2*r2; % ����ϵ��C
            D_beta=abs(C2*parent_Beta_Position(var)-parent_Position(par,var));
            X2=parent_Beta_Position(var)-A2*D_beta;
            % Delta��Hunting
            r1=rand();
            r2=rand();
            A3=2*a*r1-a; % ����ϵ��A
            C3=2*r2; % ����ϵ��C
            D_delta=abs(C3*parent_Delta_Position(var)-parent_Position(par,var));
            X3=parent_Delta_Position(var)-A3*D_delta;
            % λ�ø��£���ֹԽ��
            X=(X1+X2+X3)/3;
            X=max(X,lb(var));
            X=min(X,ub(var));
            parent_Position(par,var)=X;
        end
        parent_Val(par)=fobj(parent_Position(par,:),input_train,output_train,input_test,output_test); % �������Ŀ�꺯��ֵ
    end
    % �������죨�м��壩��Ⱥ
    for mut=1:nPop
        A=randperm(nPop); % ����˳�������������
        A(A==i)=[]; % ��ǰ��������λ���ڿգ����������м���ʱ��ǰ���岻���룩
        a=A(1);
        b=A(2);
        c=A(3);
        beta=unifrnd(beta_min,beta_max,VarSize); % ���������������
        y=parent_Position(a)+beta.*(parent_Position(b)-parent_Position(c)); % �����м���
        % ��ֹ�м���Խ��
        y=max(y,lb);
		y=min(y,ub);
        mutant_Position(mut,:)=y;
    end
    % �����Ӵ���Ⱥ��������� Crossover
    for child=1:nPop
        x=parent_Position(child,:);
        y=mutant_Position(child,:);
        z=zeros(size(x)); % ��ʼ��һ���¸���
        j0=randi([1,numel(x)]); % ����һ��α���������ѡȡ������ά�ȱ�ţ�����
        for var=1:numel(x) % ����ÿ��ά��
            if var==j0 || rand<=pCR % �����ǰά���Ǵ�����ά�Ȼ����������С�ڽ������
                z(var)=y(var); % �¸��嵱ǰά��ֵ�����м����Ӧά��ֵ
            else
                z(var)=x(var); % �¸��嵱ǰά��ֵ���ڵ�ǰ�����Ӧά��ֵ
            end
        end
        child_Position(child,:)=z; % �������֮��õ��¸���
        child_Val(child)=fobj(z,input_train,output_train,input_test,output_test); % �¸���Ŀ�꺯��ֵ
    end
    % ������Ⱥ����
    for par=1:nPop
        if child_Val(par)<parent_Val(par) % ����Ӵ��������ڸ�������
            parent_Val(par)=child_Val(par); % ���¸�������
        end
    end
    % ȷ��������Ⱥ�е�Alpha,Beta,Delta��
    [~,sort_index]=sort(parent_Val); % ������ȺĿ�꺯��ֵ����
    parent_Alpha_Position=parent_Position(sort_index(1),:); % ȷ������Alpha��
    parent_Alpha_Val=parent_Val(sort_index(1)); % ����Alpha��Ŀ�꺯��ֵ
    parent_Beta_Position=parent_Position(sort_index(2),:); % ȷ������Beta��
    parent_Delta_Position=parent_Position(sort_index(3),:); % ȷ������Delta��
    BestCost(it)=parent_Alpha_Val;
end
bestc=parent_Alpha_Position(1,1);
bestg=parent_Alpha_Position(1,2);
%% ͼʾѰ�Ź���
plot(BestCost);
xlabel('Iteration');
ylabel('Best Val');
grid on;
%% ���ûع�Ԥ�������ѵĲ�������SVM����ѵ��
cmd_cs_svr=['-s 3 -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
model_cs_svr=svmtrain(output_train,input_train,cmd_cs_svr); % SVMģ��ѵ��
%% SVM����ع�Ԥ��
[output_test_pre,~]=svmpredict(output_test,input_test,model_cs_svr); % SVMģ��Ԥ�⼰�侫��
test_pre=mapminmax('reverse',output_test_pre',rule2);
test_pre = test_pre';