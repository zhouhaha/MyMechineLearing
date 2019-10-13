for i = 1:1:50
    alpha = 0.01*i;
    M = 10000*alpha^2;
    M = round(M,0)+1;
    imfeemd = eemd(x,alpha,M);
    [m,n] = size(imfeemd);
    for j = 1:n
        for k = 1:n
            a = corrcoef(imfeemd(:,j)./x,imfeemd(:,k)./x);
            xg(j,k) = a(1,2);
        end
    end
    tmp = sum(sum(xg));
    io(i) = tmp;
end