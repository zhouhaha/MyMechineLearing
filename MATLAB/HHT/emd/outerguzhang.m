%��Ȧ��������
load 146.mat;
x=X146_DE_time;%���������ε���Ȧ����
fs=12000;%������
N=10240;%����������100����
M=0;%�������ݶε���ʼλ��
n=M:N-1;
t=n/fs;%�ź�ʱ������
X=X146_DE_time(1:N);%װ�� ���������˵���Ȧ��������
%X=X107_FE_time(1:N);%װ�� ���ȼ����˵���Ȧ��������
%X=X107_BA_time(1:N);%װ�� ���������˵���Ȧ��������
%X=X107_DE_time(1:N)-X107_BA_time(1:N);
y=X';%�źŷ�ֵ����
k_in=kurtosis(y);%�Ͷ�ϵ�����������Ϊ3����
figure;%��ԭʼ�ź�ʱ���Ƶ��ͼ
subplot(211);plot(t,y);title('ԭʼ�ź�ʱ����');
subplot(212);hua_fft(y,fs);title('ԭʼ�ź�Ƶ��');
figure;%ԭʼ�źŵİ�����
subplot(211);hua_baoluo1(y,fs,1);title('ԭʼ�źŰ�����');
subplot(212);hua_baoluo1(y,fs,1,500);title('ԭʼ�źŲ���Ƶ�ΰ�����');
imf=emd1(y);%����ģ̬�ֽ�
figure;%ǰ����IMF����
subplot(311);plot(t,imf(1,:));title('IMF1ʱ����ͼ');
subplot(312);plot(t,imf(2,:));title('IMF2ʱ����ͼ');
subplot(313);plot(t,imf(3,:));title('IMF3ʱ����ͼ');
figure;%ǰ����IMF����Ƶ��
subplot(311);hua_fft(imf(1,:),fs,1);title('IMF1Ƶ��');
subplot(312);hua_fft(imf(2,:),fs,1);title('IMF2Ƶ��');
subplot(313);hua_fft(imf(3,:),fs,1);title('IMF3Ƶ��');
figure;%ǰ����IMF����ѡ��Ƶ���ڵİ�����
xf1=0;%��Ҫ�鿴�İ�����Ƶ�ʶ����Ƶ��
xf2=1000;%��Ҫ�鿴�İ�����Ƶ�ʶ���ֹƵ��
subplot(311);hua_baol(imf(1,:),fs,1,xf1,xf2);title('IMF1������');
subplot(312);hua_baol(imf(2,:),fs,1,xf1,xf2);title('IMF2������');
subplot(313);hua_baol(imf(3,:),fs,1,xf1,xf2);title('IMF3������');

