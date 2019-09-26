%% SVR_fitness -- objective function
function fitness=fobj(cv,input_train,output_train,input_test,output_test)
% cvΪ����Ϊ2�ĺ���������SVR�в���c��v��ֵ

cmd = ['-s 3 -t 2',' -c ',num2str(cv(1)),' -g ',num2str(cv(2))];
model=svmtrain(output_train,input_train,cmd); % SVMģ��ѵ��
[~,fitness]=svmpredict(output_test,input_test,model); % SVMģ��Ԥ�⼰�侫��
fitness=fitness(2); % ��ƽ���������MSE��Ϊ�Ż���Ŀ�꺯��ֵ