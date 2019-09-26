%% SVR_fitness -- objective function
function fitness=fobj(cv,input_train,output_train,input_test,output_test)
% cv为长度为2的横向量，即SVR中参数c和v的值

cmd = ['-s 3 -t 2',' -c ',num2str(cv(1)),' -g ',num2str(cv(2))];
model=svmtrain(output_train,input_train,cmd); % SVM模型训练
[~,fitness]=svmpredict(output_test,input_test,model); % SVM模型预测及其精度
fitness=fitness(2); % 以平均均方误差MSE作为优化的目标函数值