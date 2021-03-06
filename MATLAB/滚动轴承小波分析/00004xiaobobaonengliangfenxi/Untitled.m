clear;clc;clf;
N=2048;
%fft默认计算的信号是从0开始的
t=linspace(1,2,N);deta=t(2)-t(1);fs=1/deta;%取时间长度从1到2，取频率
x=5*sin(2*pi*10*t)+5*sin(2*pi*35*t);
z=x;
c=emd(z);

%计算每个IMF分量及最后一个剩余分量residual与原始信号的相关性
[m,n]=size(c);  %m代表IMF的个数，n代表每个IMF的数据
for i=1:m;
    a=corrcoef(c(i,:),z);   %返回IMF和原信号的相关系数,a是一个2*2的矩阵，包含rxx，rxy，ryx，ryy
    xg(i)=a(1,2);
end
xg; %xg包含了所有IMF与其原有信号的相关系数

for i = 1:m
    %--------------------------------------------------------------------
    %计算各IMF的方差贡献率
    %定义：方差为平方的均值减去均值的平方
    %均值的平方
    %imfp2=mean(c(i,:),2).^2
    %平方的均值
    %imf2p=mean(c(i,:).^2,2)
    %各个IMF的方差
    mse(i) = mean(c(i,:).^2,2)-mean(c(i,:),2).^2;%求每一个imf的方差
end
mmse = sum(mse);
for i = 1:m
    mse(i) = mean(c(i,:).^2,2)-mean(c(i,:),2).^2;
    %方差百分比，也就是方差贡献率
    mseb(i)=mse(i)/mmse*100;
    %显示各个IMF的方差和贡献率
end
figure(1);
for i = 1:m
    disp(['imf',int2str(i)]);%输出语句
    disp([mse(i) mseb(i)]);
end
subplot(m+1,1,1)
plot(t,z);
set(gca,'fontname','times New Roman');
set(gca,'fontsize',14.0);
ylabel(['signal','Amplitude'])

for i=1:m
    subplot(m+1,1,i+1);
    set(gcf,'color','w')
    plot(t,c(i,:),'k')
    set(gca,'fontname','times New Roman')
    set(gca,'fontsize',14.0)
    ylabel(['imf',int2str(i)])
end


