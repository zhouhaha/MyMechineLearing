clear
clc
load('130.mat')
x = X130_BA_time;
N = length(x);
fs = 12000;
t = 0:1/fs:(N-1)/fs;
z = x
imf = emd(z);
emd_visu(z,1:length(z),imf)%显示每个IMF及参与信号
[A,f,tt] = hhspectrum(imf); %对IMF分量求取瞬时频率与振幅，A对应的振幅值，f：每个IMF对应的瞬时频率；tt：时间序列号
[im,tt,Cenf] = toimage(A,f);%对每个IMF信号合成求取hibbert谱，im:对应的振幅值，Cenf：每个网络对应的中心频率，这里横轴为时间，纵轴为频率
disp_hhs(im,tt,fs);%希尔伯特谱
colormap(flipud(gray))
%hilbert三维谱
[A,f,tt] = hhspectrum(imf);
[E,tt1]=toimage(A,f,tt,length(tt));
figure()
for i=1:size(imf,1)
faa=f(i,:);
[FA,TT1]=meshgrid(faa,tt1);%三维图显示HHT时频图
surf(FA,TT1,E)
title('HHT时频谱三维显示')
hold on
end
hold off
E=flipud(E);

