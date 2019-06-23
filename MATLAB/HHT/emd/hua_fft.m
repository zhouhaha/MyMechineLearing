%画信号的幅频谱和功率谱
%频谱使用matlab例子表示
function hua_fft(y,fs,style,varargin)%varargin表示可选多个参数
nfft= 2^nextpow2(length(y));%找出大于y的个数的最大的2的指数值（自动进算最佳FFT步长nfft）
y=y-mean(y);%去除直流分量
y_ft=fft(y,nfft);%对y信号进行DFT，得到频率的幅值分布
y_p=y_ft.*conj(y_ft)/nfft;%conj()函数是求y函数的共轭复数，实数的共轭复数是他本身,自功率谱密度函数。
y_f=fs*(0:nfft/2-1)/nfft;%在频段上取频率的范围作为横坐标
y_p=y_ft.*conj(y_ft)/nfft;%conj()函数是求y函数的共轭复数，实数的共轭复数是他本身。
if style==1
    if nargin==3
        plot(y_f,2*abs(y_ft(1:nfft/2))/length(y));%matlab的帮助里画FFT的方法
        ylabel('幅值');xlabel('频率');title('信号幅值谱');
        plot(y_f,abs(y_ft(1:nfft/2)));%论坛上画FFT的方法
    else
        f1=varargin{1};%可选多个参数中的第一个
        fn=varargin{2};
        ni=round(f1 * nfft/fs+1);
        na=round(fn * nfft/fs+1);
        plot(y_f(ni:na),abs(y_ft(ni:na)*2/nfft));
    end
elseif style==2
    plot(y_f,y_p(1:nfft/2));
    ylabel('功率谱密度');xlabel('频率');title('信号功率谱');
else
    subplot(211);plot(y_f,2*abs(y_ft(1:nfft/2))/length(y));
    ylabel('幅值');xlabel('频率');title('信号幅值谱');
    subplot(212);plot(y_f,y_p(1:nfft/2));
    ylabel('功率谱密度');xlabel('频率');title('信号功率谱');
end
end
