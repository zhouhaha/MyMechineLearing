clear all;
close all;
clc;
% Time Domain 0 to T
T = 1000;
fs = 1/T;
t = (1:T)/T;
freqs = 2*pi*(t-0.5-1/T)/(fs);
% center frequencies of components
f_1 = 2;
f_2 = 24;
f_3 = 288;
% modes
v_1 = (cos(2*pi*f_1*t));
v_2 = 1/4*(cos(2*pi*f_2*t));
v_3 = 1/16*(cos(2*pi*f_3*t));
% for visualization purposes
wsub{1} = 2*pi*f_1;
wsub{2} = 2*pi*f_2;
wsub{3} = 2*pi*f_3;
% composite signal, including noise
f = v_1 + v_2 + v_3 + 0.1*randn(size(v_1));
% some sample parameters for VMD
alpha = 2000;        % moderate bandwidth constraint
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
K = 4;              % 4 modes
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 1e-7;
 
%--------------- Run actual VMD code
[u, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
subplot(size(u,1)+1,2,1);
plot(t,f,'k');grid on;
title('VMD分解');
subplot(size(u,1)+1,2,2);
plot(freqs,abs(fft(f)),'k');grid on;
title('对应频谱');
for i = 2:size(u,1)+1
    subplot(size(u,1)+1,2,i*2-1);
    plot(t,u(i-1,:),'k');grid on;
    subplot(size(u,1)+1,2,i*2);
    plot(freqs,abs(fft(u(i-1,:))),'k');grid on;
end