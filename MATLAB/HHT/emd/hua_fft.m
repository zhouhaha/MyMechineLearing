%���źŵķ�Ƶ�׺͹�����
%Ƶ��ʹ��matlab���ӱ�ʾ
function hua_fft(y,fs,style,varargin)%varargin��ʾ��ѡ�������
nfft= 2^nextpow2(length(y));%�ҳ�����y�ĸ���������2��ָ��ֵ���Զ��������FFT����nfft��
y=y-mean(y);%ȥ��ֱ������
y_ft=fft(y,nfft);%��y�źŽ���DFT���õ�Ƶ�ʵķ�ֵ�ֲ�
y_p=y_ft.*conj(y_ft)/nfft;%conj()��������y�����Ĺ������ʵ���Ĺ������������,�Թ������ܶȺ�����
y_f=fs*(0:nfft/2-1)/nfft;%��Ƶ����ȡƵ�ʵķ�Χ��Ϊ������
y_p=y_ft.*conj(y_ft)/nfft;%conj()��������y�����Ĺ������ʵ���Ĺ������������
if style==1
    if nargin==3
        plot(y_f,2*abs(y_ft(1:nfft/2))/length(y));%matlab�İ����ﻭFFT�ķ���
        ylabel('��ֵ');xlabel('Ƶ��');title('�źŷ�ֵ��');
        plot(y_f,abs(y_ft(1:nfft/2)));%��̳�ϻ�FFT�ķ���
    else
        f1=varargin{1};%��ѡ��������еĵ�һ��
        fn=varargin{2};
        ni=round(f1 * nfft/fs+1);
        na=round(fn * nfft/fs+1);
        plot(y_f(ni:na),abs(y_ft(ni:na)*2/nfft));
    end
elseif style==2
    plot(y_f,y_p(1:nfft/2));
    ylabel('�������ܶ�');xlabel('Ƶ��');title('�źŹ�����');
else
    subplot(211);plot(y_f,2*abs(y_ft(1:nfft/2))/length(y));
    ylabel('��ֵ');xlabel('Ƶ��');title('�źŷ�ֵ��');
    subplot(212);plot(y_f,y_p(1:nfft/2));
    ylabel('�������ܶ�');xlabel('Ƶ��');title('�źŹ�����');
end
end
