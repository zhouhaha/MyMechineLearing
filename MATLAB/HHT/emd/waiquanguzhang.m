clear all;
close all;
load 130.mat;%外圈故障数据
x=X130_FE_time;%驱动计数段的内圈故障
fs=12000;%采样率
N=10240;%采样点数（100倍）
M=0;%采样数据段的起始位置
n=M:N-1;
t=n/fs;%信号时间序列
%X=X130_DE_time(1:N);%装载 驱动计数端的内圈故障数据
X=X130_FE_time(1:N);%装载 风扇计数端的内圈故障数据
y=X';%信号幅值序列
imf=emd(y);%经验模态分解                                     !!!
%X=X130_BA_time(1:N);%装载 基础计数端的内圈故障数据
Z=X130_FE_time(1:N)-imf(1,N)-imf(2,N)-imf(3,N);
z=Z';%去噪
k_in=kurtosis(y);%峭度系数，正常轴承为3左右
figure;%画原始信号时域和频域图
subplot(211);plot(t,y);title('原始信号时域波形');
subplot(212);hua_fft(y,fs,1);title('原始信号频谱');
figure;%原始信号的包络谱
subplot(211);hua_baoluo(y,fs,1);title('原始信号包络谱');
subplot(212);hua_baoluo(y,fs,1,1,1000);title('原始信号部分频段包络谱');%!!!
%subplot(212);hua_baoluo(y,fs,1);axis([0 400 0 0.4]);title('原始信号部分频段包络谱');
figure;%前三个IMF分量
subplot(311);plot(t,imf(1,:));title('IMF1时域波形图');
subplot(312);plot(t,imf(2,:));title('IMF2时域波形图');
subplot(313);plot(t,imf(3,:));title('IMF3时域波形图');
figure;%前三个IMF分量频谱
subplot(311);hua_fft(imf(1,:),fs,1);title('IMF1频谱');
subplot(312);hua_fft(imf(2,:),fs,1);title('IMF2频谱');%幅值与频率的关系
subplot(313);hua_fft(imf(3,:),fs,1);title('IMF3频谱');
figure;%前三个IMF分量选择频段内的包络谱
xf1=0;%需要查看的包络谱频率段起点频率
xf2=1000;%需要查看的包络谱频率段终止频率
subplot(311);hua_baoluo(imf(1,:),fs,1,xf1,xf2);title('IMF1包络谱');
subplot(312);hua_baoluo(imf(2,:),fs,1,xf1,xf2);title('IMF2包络谱');%取各频率的振幅最大值进行包络形成的谱
subplot(313);hua_baoluo(imf(3,:),fs,1,xf1,xf2);title('IMF3包络谱');
figure;
plot(t,z);title('原始信号去噪值');


