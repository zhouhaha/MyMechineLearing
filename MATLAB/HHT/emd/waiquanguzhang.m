clear all;
close all;
load 130.mat;%��Ȧ��������
x=X130_FE_time;%���������ε���Ȧ����
fs=12000;%������
N=10240;%����������100����
M=0;%�������ݶε���ʼλ��
n=M:N-1;
t=n/fs;%�ź�ʱ������
%X=X130_DE_time(1:N);%װ�� ���������˵���Ȧ��������
X=X130_FE_time(1:N);%װ�� ���ȼ����˵���Ȧ��������
y=X';%�źŷ�ֵ����
imf=emd(y);%����ģ̬�ֽ�                                     !!!
%X=X130_BA_time(1:N);%װ�� ���������˵���Ȧ��������
Z=X130_FE_time(1:N)-imf(1,N)-imf(2,N)-imf(3,N);
z=Z';%ȥ��
k_in=kurtosis(y);%�Ͷ�ϵ�����������Ϊ3����
figure;%��ԭʼ�ź�ʱ���Ƶ��ͼ
subplot(211);plot(t,y);title('ԭʼ�ź�ʱ����');
subplot(212);hua_fft(y,fs,1);title('ԭʼ�ź�Ƶ��');
figure;%ԭʼ�źŵİ�����
subplot(211);hua_baoluo(y,fs,1);title('ԭʼ�źŰ�����');
subplot(212);hua_baoluo(y,fs,1,1,1000);title('ԭʼ�źŲ���Ƶ�ΰ�����');%!!!
%subplot(212);hua_baoluo(y,fs,1);axis([0 400 0 0.4]);title('ԭʼ�źŲ���Ƶ�ΰ�����');
figure;%ǰ����IMF����
subplot(311);plot(t,imf(1,:));title('IMF1ʱ����ͼ');
subplot(312);plot(t,imf(2,:));title('IMF2ʱ����ͼ');
subplot(313);plot(t,imf(3,:));title('IMF3ʱ����ͼ');
figure;%ǰ����IMF����Ƶ��
subplot(311);hua_fft(imf(1,:),fs,1);title('IMF1Ƶ��');
subplot(312);hua_fft(imf(2,:),fs,1);title('IMF2Ƶ��');%��ֵ��Ƶ�ʵĹ�ϵ
subplot(313);hua_fft(imf(3,:),fs,1);title('IMF3Ƶ��');
figure;%ǰ����IMF����ѡ��Ƶ���ڵİ�����
xf1=0;%��Ҫ�鿴�İ�����Ƶ�ʶ����Ƶ��
xf2=1000;%��Ҫ�鿴�İ�����Ƶ�ʶ���ֹƵ��
subplot(311);hua_baoluo(imf(1,:),fs,1,xf1,xf2);title('IMF1������');
subplot(312);hua_baoluo(imf(2,:),fs,1,xf1,xf2);title('IMF2������');%ȡ��Ƶ�ʵ�������ֵ���а����γɵ���
subplot(313);hua_baoluo(imf(3,:),fs,1,xf1,xf2);title('IMF3������');
figure;
plot(t,z);title('ԭʼ�ź�ȥ��ֵ');


