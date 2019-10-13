function M=mymape(actualdata,predata)
M=mean(abs((actualdata-predata)./actualdata));