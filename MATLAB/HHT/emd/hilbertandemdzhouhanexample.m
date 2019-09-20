clear
clc
load('130.mat')
x = X130_BA_time;
N = length(x);
fs = 12000;
t = 0:1/fs:(N-1)/fs;
z = x
imf = emd(z);
emd_visu(z,1:length(z),imf)
[A,f,tt] = hhspectrum(imf);
[im,tt] = toimage(A,f);
disp_hhs(im,[],fs);
colormap(flipud(gray))
