load 130.mat;%内圈故障数据
x=X107_DE_time;%驱动计数段的内圈故障
fs=12000;%采样率
N=10240;%采样点数（100倍）
M=0;%采样数据段的起始位置
n=M:N-1;
t=n/fs;%信号时间序列
X=X107_DE_time(1:N);%装载 驱动计数端的内圈故障数据
%X=X107_FE_time(1:N);%装载 风扇计数端的内圈故障数据
%X=X107_BA_time(1:N);%装载 基础计数端的内圈故障数据
%X=X107_DE_time(1:N)-X107_BA_time(1:N);
y=X';%信号幅值序列
% k_in=kurtosis(y);%峭度系数，正常轴承为3左右
  subplot(212);hua_fft(y,fs,1);title('原始信号频谱');
%figure;%原始信号的包络谱
%subplot(211);hua_baoluo(y,fs,1);title('原始信号包络谱');
% subplot(212);hua_baoluo(y,fs,1,500);title('原始信号部分频段包络谱');%!!!
%subplot(212);hua_baoluo(y,fs,1);axis([0 400 0 0.4]);title('原始信号部分频段包络谱');
imf=emd(y);%经验模态分解                                     !!!
fi gure;%前三个IMF分量
subplot(311);plot(t,imf(1,:));title('IMF1时域波形图');
 subplot(313);plot(t,imf(3,:));title('IMF3时域波形图');
figure;%前三个IMF分量频谱
subplot(311);hua_fft(imf(1,:),fs,1);title('IMF1频谱');
subplot(312);hua_fft(imf(2,:),fs,1);title('IMF2频谱');
subplot(313);hua_fft(imf(3,:),fs,1);title('IMF3频谱');
figure;%前三个IMF分量选择频段内的包络谱
xf1=0;%需要查看的包络谱频率段起点频率
xf2=1000;%需要查看的包络谱频率段终止频率 
subplot(311);hua_baoluo(imf(1,:),fs,1,xf1,xf2);title('IMF1包络谱');
subplot(312);hua_baoluo(imf(2,:),fs,1,xf1,xf2);title('IMF2包络谱');
subplot(313);hua_baoluo(imf(3,:),fs,1,xf1,xf2);title('IMF3包络谱');

