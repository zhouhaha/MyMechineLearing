x=@(t) (1+0.5*cos(9*pi*t)).*cos(200*pi*t+2*cos(10*pi*t))+sin(pi*t).*sin(30*pi*t);
t=0:0.01:9.99;
fs=1000;
y=x(t);
subplot(311);plot(t,y);
subplot(312);hua_fft(y,fs,1);%ֱ�ӻ�����ֵ��
subplot(313);hua_fft(y,fs,1,0,250);%ֱ�ӻ�����ֵ�ף�����ֻ����0-250Hz��Χ�ķ�ֵ��
figure;
subplot(311);plot(t,y);
subplot(312);hua_baol(y,fs,1);%ֱ�ӻ���������
subplot(313);hua_baol(y,fs,1,0,250);%ֱ�ӻ��������ף�����ֻ����0-250Hz��Χ�İ���
