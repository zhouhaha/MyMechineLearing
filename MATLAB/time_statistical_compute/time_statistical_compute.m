function time=time_statistical_compute(x)
%%对时域信号进行统计量分析
%% p2,p10返回有量纲指标，f1,f2,f3,f4,f5返回无量纲指标
N=length(x);
p1=mean(x); %均值
x=x-p1;
p2=sqrt(sum(x.^2)/N); %均方根值,又称有效值(!)
p3=(sum(sqrt(abs(x)))/N).^2; %方根幅值(!)
p4=sum(abs(x))/N; %绝对平均值
p5=sum(x.^3)/N; %歪度
p6=sum(x.^4)/N; %峭度
p7=sum((x).^2)/N; %方差
p8=max(x);%最大值
p9=min(x);%最小值
p10=p8-p9;%峰峰值
p11=
%%以上都是有量纲统计量，以下是无量纲统计量
f1=p2/p4; %波形指标
f2=p8/p2; %峰值指标  
f3=p8/p4; %脉冲指标
f4=p8/p3; %裕度指标
f5=p6/((p2)^4); %峭度指标
time=[p1,p2,p3,p4,p5,p6,p7,p10,f1,f2,f3,f4,f5];
