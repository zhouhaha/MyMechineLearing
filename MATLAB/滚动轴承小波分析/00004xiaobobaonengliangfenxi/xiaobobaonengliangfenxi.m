%%%%������ȡС�����ع��źŵĹ������
clc
clear
fs=12e3;
load zhengchang.mat;%���������ź�
load 98.mat;
s1=X098_DE_time(19001:1:20000,1);% �������������������
% s1=X098_FE_time(1:1:10000,1);% �������������������
load in.mat;%������Ȧ�����ź�
% load fan_in.mat;%���ط��ȶ���Ȧ�����ź�
load 106.mat;
s2=X106_DE_time(19001:1:20000,1);% ��Ȧ�����ź����������������
% s2=X279_FE_time(1:1:10000,1);% ��Ȧ�����ź����������������
% load fan_ball.mat;%���ع���������ź�
load ball.mat;%���ع���������ź�
load 119.mat;
s3=X119_DE_time(19001:1:20000,1);% ������������������������
% s3=X283_FE_time(1:1:10000,1);% ������������������������
% load fan_out_12.mat;%������Ȧ12��λ�ù����ź�
load out_12.mat;%������Ȧ12��λ�ù����ź�
load 158.mat;
s4=X158_DE_time(14001:1:15000,1);% ��Ȧ12��λ�����������������
% s4=X305_FE_time(1:1:10000,1);% ��Ȧ12��λ�����������������
n=3;
ji='db5';
wpt=wpdec(s1,n,ji);%ʹ��db5С�����ֽ������źŵ������㣬ʹ��shannon��;���²����ֽ�֮�������Ϊ8����rcfs10-rcfs17;s1��ʱ��ͼ��Ϊ��˸�������Ӽ��ɵõ� 
% plot(wpt);
rcfs10=wprcoef(wpt,[n,0]);%������Ӧ��Ƶ�ν����ع�������С������T�Ľڵ�N���ع�ϵ��;
rcfs11=wprcoef(wpt,[n,1]);
rcfs12=wprcoef(wpt,[n,2]);
rcfs13=wprcoef(wpt,[n,3]);
rcfs14=wprcoef(wpt,[n,4]);
rcfs15=wprcoef(wpt,[n,5]);
rcfs16=wprcoef(wpt,[n,6]);
rcfs17=wprcoef(wpt,[n,7]);
cfs10=wpcoef(wpt,[n,0]);%��ȡС�����ֽ�֮��İ˸�С����ϵ��
cfs11=wpcoef(wpt,[n,1]);
cfs12=wpcoef(wpt,[n,2]);
cfs13=wpcoef(wpt,[n,3]);
cfs14=wpcoef(wpt,[n,4]);
cfs15=wpcoef(wpt,[n,5]);
cfs16=wpcoef(wpt,[n,6]);
cfs17=wpcoef(wpt,[n,7]);
wpt1=wpdec(s2,n,ji);
rcfs20=wprcoef(wpt1,[n,0]);
rcfs21=wprcoef(wpt1,[n,1]);
rcfs22=wprcoef(wpt1,[n,2]);
rcfs23=wprcoef(wpt1,[n,3]);
rcfs24=wprcoef(wpt1,[n,4]);
rcfs25=wprcoef(wpt1,[n,5]);
rcfs26=wprcoef(wpt1,[n,6]);
rcfs27=wprcoef(wpt1,[n,7]);
cfs20=wpcoef(wpt1,[n,0]);
cfs21=wpcoef(wpt1,[n,1]);
cfs22=wpcoef(wpt1,[n,2]);
cfs23=wpcoef(wpt1,[n,3]);
cfs24=wpcoef(wpt1,[n,4]);
cfs25=wpcoef(wpt1,[n,5]);
cfs26=wpcoef(wpt1,[n,6]);
cfs27=wpcoef(wpt1,[n,7]);
wpt2=wpdec(s3,n,ji);
rcfs30=wprcoef(wpt2,[n,0]);
rcfs31=wprcoef(wpt2,[n,1]);
rcfs32=wprcoef(wpt2,[n,2]);
rcfs33=wprcoef(wpt2,[n,3]);
rcfs34=wprcoef(wpt2,[n,4]);
rcfs35=wprcoef(wpt2,[n,5]);
rcfs36=wprcoef(wpt2,[n,6]);
rcfs37=wprcoef(wpt2,[n,7]);
cfs30=wpcoef(wpt2,[n,0]);
cfs31=wpcoef(wpt2,[n,1]);
cfs32=wpcoef(wpt2,[n,2]);
cfs33=wpcoef(wpt2,[n,3]);
cfs34=wpcoef(wpt2,[n,4]);
cfs35=wpcoef(wpt2,[n,5]);
cfs36=wpcoef(wpt2,[n,6]);
cfs37=wpcoef(wpt2,[n,7]);
wpt3=wpdec(s4,n,ji);
rcfs40=wprcoef(wpt3,[n,0]);
rcfs41=wprcoef(wpt3,[n,1]);
rcfs42=wprcoef(wpt3,[n,2]);
rcfs43=wprcoef(wpt3,[n,3]);
rcfs44=wprcoef(wpt3,[n,4]);
rcfs45=wprcoef(wpt3,[n,5]);
rcfs46=wprcoef(wpt3,[n,6]);
rcfs47=wprcoef(wpt3,[n,7]);
cfs40=wpcoef(wpt3,[n,0]);
cfs41=wpcoef(wpt3,[n,1]);
cfs42=wpcoef(wpt3,[n,2]);
cfs43=wpcoef(wpt3,[n,3]);
cfs44=wpcoef(wpt3,[n,4]);
cfs45=wpcoef(wpt3,[n,5]);
cfs46=wpcoef(wpt3,[n,6]);
cfs47=wpcoef(wpt3,[n,7]);
for i=1:2^n
E(i)=(norm(wpcoef(wpt,[n,i-1]),2))^2;%���i���ڵ��2������ƽ������ʵҲ����ƽ���ͣ�������
end
E_total=sum(E); %��������
for i=1:2^n
pfir(i)= E(i)/E_total;%����ÿ���ڵ�ĸ��ʣ�����ռ�������ȣ���ΪE(i)/E_total
end

for i=1:2^n
E1(i)=(norm(wpcoef(wpt1,[n,i-1]),2))^2;%���i���ڵ�ķ���ƽ������ʵҲ����ƽ���ͣ�������
end
E1_total=sum(E1); %��������
for i=1:2^n
pfir1(i)= E1(i)/E1_total;%����ÿ���ڵ�ĸ��ʣ�����ռ��������
end

for i=1:2^n
E2(i)=(norm(wpcoef(wpt2,[n,i-1]),2))^2;
end
E2_total=sum(E2);
for i=1:2^n
pfir2(i)= E2(i)/E2_total;
end

for i=1:2^n
E3(i)=(norm(wpcoef(wpt3,[n,i-1]),2))^2;
end
E3_total=sum(E3);
for i=1:2^n
pfir3(i)= E3(i)/E3_total;
end
%%%%������������ź��������Ȧ�����źŵ��ع�
figure(1)
subplot(9,2,1)
plot(s1)
title('��������źţ�Ƶ�ο��12000/��2*8����')
ylabel('S1')

subplot(9,2,2)
plot(s2)
title('�����Ȧ�����źţ�Ƶ�ο��12000/��2*8����')
ylabel('S2')

subplot(9,2,3)
plot(rcfs10)
ylabel('S130')

subplot(9,2,5)
plot(rcfs11)
ylabel('S131')

subplot(9,2,7)
plot(rcfs12)
ylabel('S132')

subplot(9,2,9)
plot(rcfs13)
ylabel('S133')

subplot(9,2,11)
plot(rcfs14)
ylabel('S134')

subplot(9,2,13)
plot(rcfs15)
ylabel('S135')

subplot(9,2,15)
plot(rcfs16)
ylabel('S136')

subplot(9,2,17)
plot(rcfs17)
ylabel('S137')
subplot(9,2,4)
plot(rcfs20)
ylabel('S230')
subplot(9,2,6)
plot(rcfs21)
ylabel('S231')
subplot(9,2,8)
plot(rcfs22)
ylabel('S232')
subplot(9,2,10)
plot(rcfs23)
ylabel('S233')
subplot(9,2,12)
plot(rcfs24)
ylabel('S234')
subplot(9,2,14)
plot(rcfs25)
ylabel('S235')
subplot(9,2,16)
plot(rcfs26)
ylabel('S236')
subplot(9,2,18)
plot(rcfs27)
ylabel('S237')
%%%%%%%%������й�������Ϻ���Ȧ�����ź��ع�ͼ
figure(2)
subplot(9,2,1)%9��2�еĵ�һ��
plot(s3)
title('��й���������źţ�Ƶ�ο��12000/��2*8����')
ylabel('S3')
subplot(9,2,2)%9��2�еĵڶ���
plot(s4)
title('�����Ȧ�����źţ�Ƶ�ο��12000/��2*8����')
ylabel('S4')
subplot(9,2,3)
plot(rcfs30)
ylabel('S330')
subplot(9,2,5)
plot(rcfs31)
ylabel('S331')
subplot(9,2,7)
plot(rcfs32)
ylabel('S332')
xlabel('ʱ��')
subplot(9,2,9)
plot(rcfs33)
ylabel('S333')
subplot(9,2,11)
plot(rcfs34)
ylabel('S334')
subplot(9,2,13)
plot(rcfs35)
ylabel('S335')
subplot(9,2,15)
plot(rcfs36)
ylabel('S336')
subplot(9,2,17)
plot(rcfs37)
ylabel('S337')
subplot(9,2,4)
plot(rcfs40)
ylabel('S430')
subplot(9,2,6)
plot(rcfs41)
ylabel('S431')
subplot(9,2,8)
plot(rcfs42)
ylabel('S432')
subplot(9,2,10)
plot(rcfs43)
ylabel('S433')
subplot(9,2,12)
plot(rcfs44)
ylabel('S434')
subplot(9,2,14)
plot(rcfs45)
ylabel('S435')
subplot(9,2,16)
plot(rcfs46)
ylabel('S436')
subplot(9,2,18)
plot(rcfs47)
ylabel('S437')
%%%%%%%%%%%��������ͼ
figure(3)
subplot(2,2,1)
bar(pfir)%%����������
title('��������ź�������')
subplot(2,2,2)
bar(pfir1)%%����������
title('�����Ȧ�����ź�������')
subplot(2,2,3)
bar(pfir2)%%����������
title('��й���������ź�������')
subplot(2,2,4)
bar(pfir3)%%����������
title('�����Ȧ�����ź�������')
%%%%%%%%%%%%%%%%%%%%%
 %plot your figure before
 %%%%%%%%%%%%%%%%%%%%%
 % figure resize
set(gcf,'Position',[100 100 260 220]);%��������û�ͼ�Ĵ�С������Ҫ��word���ٵ�����С���Ҹ��Ĳ�����ͼ�Ĵ�С��7cm
% set(gca,'Position',[.13 .17 .80 .74]);%���������xy����ͼƬ��ռ�ı�����������Ҫ�Լ�΢����
figure_FontSize=12;
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
% set(get(gca,'Title'),'FontSize',figure_FontSize,'Vertical','middle');
% set(get(gca,'TextBox'),'FontSize',figure_FontSize,'Vertical','middle');
set(findobj('FontSize',12),'FontSize',figure_FontSize);%��6���ǽ������С��Ϊ8���֣���Сͼ�������
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',1.4);%����ǽ��߿��Ϊ2

 


% figure(3)
% subplot(1,2,1);
% E1= wenergy(wpt);
% E11=E1'
% bar(E1)
% subplot(1,2,2);
% E2= wenergy(wpt1);
% E12=E2'
% bar(E2)