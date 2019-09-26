clear
clc
close all
load wndspd % 示例数据为风速（时间序列）数据，共144个样本
%% HGWO-SVR
% 训练/测试数据准备（用前3天预测后一天）,用前100天做训练数据
input_train(1,:)=wndspd(1:97);
input_train(2,:)=wndspd(2:98);
input_train(3,:)=wndspd(3:99);
output_train=[wndspd(4:100)]';
input_test(1,:)=wndspd(101:end-3);
input_test(2,:)=wndspd(102:end-2);
input_test(3,:)=wndspd(103:end-1);
output_test=(wndspd(104:end))';
para=[30,500,0.2,0.8,0.2];
tic
[bestc,bestg,test_pre]=my_HGWO_SVR(para,input_train',output_train',input_test',output_test');
toc
%% 预测结果图
err_pre=output_test'-test_pre;
figure('Name','测试数据残差图')
set(gcf,'unit','centimeters','position',[0.5,5,30,5])
plot(err_pre,'*-');
figure('Name','原始-预测图')
plot(test_pre,'*r-');hold on;plot(output_test,'bo-');
legend('预测','原始')
set(gcf,'unit','centimeters','position',[0.5,13,30,5])
