%%%%以下提取小波包重构信号的故障诊断
clc
clear
fs=12e3;
load zhengchang.mat;%加载正常信号
load 98.mat;
s1=X098_DE_time(19001:1:20000,1);% 正常情况下驱动端数据
% s1=X098_FE_time(1:1:10000,1);% 正常情况下驱动端数据
load in.mat;%加载内圈故障信号
% load fan_in.mat;%加载风扇端内圈故障信号
load 106.mat;
s2=X106_DE_time(19001:1:20000,1);% 内圈故障信号情况下驱动端数据
% s2=X279_FE_time(1:1:10000,1);% 内圈故障信号情况下驱动端数据
% load fan_ball.mat;%加载滚动体故障信号
load ball.mat;%加载滚动体故障信号
load 119.mat;
s3=X119_DE_time(19001:1:20000,1);% 滚动体故障情况下驱动端数据
% s3=X283_FE_time(1:1:10000,1);% 滚动体故障情况下驱动端数据
% load fan_out_12.mat;%加载外圈12点位置故障信号
load out_12.mat;%加载外圈12点位置故障信号
load 158.mat;
s4=X158_DE_time(14001:1:15000,1);% 外圈12点位置情况下驱动端数据
% s4=X305_FE_time(1:1:10000,1);% 外圈12点位置情况下驱动端数据
n=3;
ji='db5';
wpt=wpdec(s1,n,ji);%使用db5小波包分解正常信号到底三层，使用shannon熵;下坡播报分解之后的数据为8个：rcfs10-rcfs17;s1的时域图形为这八个数据相加即可得到 
% plot(wpt);
rcfs10=wprcoef(wpt,[n,0]);%利用相应的频段进行重构；计算小波包树T的节点N的重构系数;
rcfs11=wprcoef(wpt,[n,1]);
rcfs12=wprcoef(wpt,[n,2]);
rcfs13=wprcoef(wpt,[n,3]);
rcfs14=wprcoef(wpt,[n,4]);
rcfs15=wprcoef(wpt,[n,5]);
rcfs16=wprcoef(wpt,[n,6]);
rcfs17=wprcoef(wpt,[n,7]);
cfs10=wpcoef(wpt,[n,0]);%提取小波包分解之后的八个小波的系数
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
E(i)=(norm(wpcoef(wpt,[n,i-1]),2))^2;%求第i个节点的2范数的平方，其实也就是平方和，即能量
end
E_total=sum(E); %求总能量
for i=1:2^n
pfir(i)= E(i)/E_total;%若求每个节点的概率，即所占的能量比，则为E(i)/E_total
end

for i=1:2^n
E1(i)=(norm(wpcoef(wpt1,[n,i-1]),2))^2;%求第i个节点的范数平方，其实也就是平方和，即能量
end
E1_total=sum(E1); %求总能量
for i=1:2^n
pfir1(i)= E1(i)/E1_total;%若求每个节点的概率，即所占的能量比
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
%%%%绘制轴承正常信号与轴承内圈故障信号的重构
figure(1)
subplot(9,2,1)
plot(s1)
title('轴承正常信号（频段宽度12000/（2*8））')
ylabel('S1')

subplot(9,2,2)
plot(s2)
title('轴承内圈故障信号（频段宽度12000/（2*8））')
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
%%%%%%%%绘制轴承滚动体故障和外圈故障信号重构图
figure(2)
subplot(9,2,1)%9行2列的第一个
plot(s3)
title('轴承滚动体故障信号（频段宽度12000/（2*8））')
ylabel('S3')
subplot(9,2,2)%9行2列的第二个
plot(s4)
title('轴承外圈故障信号（频段宽度12000/（2*8））')
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
xlabel('时间')
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
%%%%%%%%%%%绘制能量图
figure(3)
subplot(2,2,1)
bar(pfir)%%计算能量谱
title('轴承正常信号能量谱')
subplot(2,2,2)
bar(pfir1)%%计算能量谱
title('轴承内圈故障信号能量谱')
subplot(2,2,3)
bar(pfir2)%%计算能量谱
title('轴承滚动体故障信号能量谱')
subplot(2,2,4)
bar(pfir3)%%计算能量谱
title('轴承外圈故障信号能量谱')
%%%%%%%%%%%%%%%%%%%%%
 %plot your figure before
 %%%%%%%%%%%%%%%%%%%%%
 % figure resize
set(gcf,'Position',[100 100 260 220]);%这句是设置绘图的大小，不需要到word里再调整大小。我给的参数，图的大小是7cm
% set(gca,'Position',[.13 .17 .80 .74]);%这句是设置xy轴在图片中占的比例，可能需要自己微调。
figure_FontSize=12;
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
% set(get(gca,'Title'),'FontSize',figure_FontSize,'Vertical','middle');
% set(get(gca,'TextBox'),'FontSize',figure_FontSize,'Vertical','middle');
set(findobj('FontSize',12),'FontSize',figure_FontSize);%这6句是将字体大小改为8号字，在小图里很清晰
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',1.4);%这句是将线宽改为2

 


% figure(3)
% subplot(1,2,1);
% E1= wenergy(wpt);
% E11=E1'
% bar(E1)
% subplot(1,2,2);
% E2= wenergy(wpt1);
% E12=E2'
% bar(E2)