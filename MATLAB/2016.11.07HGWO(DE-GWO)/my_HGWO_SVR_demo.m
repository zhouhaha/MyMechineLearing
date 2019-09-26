clear
clc
close all
load wndspd % ʾ������Ϊ���٣�ʱ�����У����ݣ���144������
%% HGWO-SVR
% ѵ��/��������׼������ǰ3��Ԥ���һ�죩,��ǰ100����ѵ������
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
%% Ԥ����ͼ
err_pre=output_test'-test_pre;
figure('Name','�������ݲв�ͼ')
set(gcf,'unit','centimeters','position',[0.5,5,30,5])
plot(err_pre,'*-');
figure('Name','ԭʼ-Ԥ��ͼ')
plot(test_pre,'*r-');hold on;plot(output_test,'bo-');
legend('Ԥ��','ԭʼ')
set(gcf,'unit','centimeters','position',[0.5,13,30,5])
