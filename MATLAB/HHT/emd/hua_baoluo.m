function hua_baoluo(y,fs,style,varargin)
%�����纯����hua_baoluo�Ĵ���򻯰�
%�������Ҫô��3����Ҫô��5��
%��������ʱ��
y_hht=hilbert(y);%ϣ�����ر任
y_an=abs(y_hht);%�����ź�
y_an=y_an-mean(y_an);%ȥ��ֱ������
if nargin==3    %�����������������
    hua_fft(y_an,fs,style);
elseif nargin==5
    f1=varargin{1};
    f2=varargin{2};
    hua_fft(y_an,fs,style,f1,f2);
else
    error('���ú��������������Ŀ����ȷ���������ֻ���������������');
end
end
