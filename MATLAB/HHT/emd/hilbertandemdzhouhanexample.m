clear
clc
load('130.mat')
x = X130_BA_time;
N = length(x);
fs = 12000;
t = 0:1/fs:(N-1)/fs;
z = x
imf = emd(z);
emd_visu(z,1:length(z),imf)%��ʾÿ��IMF�������ź�
[A,f,tt] = hhspectrum(imf); %��IMF������ȡ˲ʱƵ���������A��Ӧ�����ֵ��f��ÿ��IMF��Ӧ��˲ʱƵ�ʣ�tt��ʱ�����к�
[im,tt,Cenf] = toimage(A,f);%��ÿ��IMF�źźϳ���ȡhibbert�ף�im:��Ӧ�����ֵ��Cenf��ÿ�������Ӧ������Ƶ�ʣ��������Ϊʱ�䣬����ΪƵ��
disp_hhs(im,tt,fs);%ϣ��������
colormap(flipud(gray))
%hilbert��ά��
[A,f,tt] = hhspectrum(imf);
[E,tt1]=toimage(A,f,tt,length(tt));
figure()
for i=1:size(imf,1)
faa=f(i,:);
[FA,TT1]=meshgrid(faa,tt1);%��άͼ��ʾHHTʱƵͼ
surf(FA,TT1,E)
title('HHTʱƵ����ά��ʾ')
hold on
end
hold off
E=flipud(E);

