%%输入参数的定义如下：
%imf:输入的imf分解函数
%numfreq:numfreq：划分的频率个数
%fs：采样频率
%ANGLE：视图角度
function plot_hht_3d(imf,numfreq,fs,ANGLE)
if nargin<3
    fs=1;
    ANGLE=[-37.5,30];
end
if nargin<4
    if size(fs,2)>1
        ANGLE=fs;
        fs=1;
    else
        ANGLE=[-37.5,30];
    end   
end
N=size(imf,2);
[A,f,tt]=hhspectrum(imf);
[m,n]=size(f);
MaxFreq=max(max(f));
MaxFreq=ceil(MaxFreq/0.5)*0.5;
if nargin<2
    numfreq=512;
end
df=linspace(0,MaxFreq,numfreq);
Spectrum=zeros(numfreq,n);
Temp=f;
Temp=min(round((Temp/MaxFreq)*numfreq)+1,numfreq);
for k=1:m
    for u=1:n
        Spectrum(Temp(k,u),u)=Spectrum(Temp(k,u),u)+A(k,u);
    end
end
df=df*fs;
figure
clf
mesh(tt,df,Spectrum);
set(gca,'XLim',[0,N/fs]);
xlabel('时间/s');
xlabel('采样点数/n');
if fs==1
    ylabel('归一化频率');
else
    ylabel('频率/Hz');
end
zlabel('幅值');
title('三维联合时频图');
colormap jet;
shading interp;
view(ANGLE(1),ANGLE(2));
set(gca,'YLim',[0,fs/2]);
end