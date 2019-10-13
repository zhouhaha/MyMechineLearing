function M=mymse(actualdata,predata)
M=mean(power(actualdata-predata,2));