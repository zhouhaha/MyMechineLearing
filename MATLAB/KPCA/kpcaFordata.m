function [P,mpIndex,Result] = kpcaFordata(X,c,threshold) %
%load tsstdata  %所导入的数据
%X = testdata(:,1:4)'; % X训练数据集，数据的行为特征个数，列为一个样本
%%将数据归一化，归一化为均值为0，方差为1
[Xrow, Xcol] = size(X); % Xrow：样本个数 Xcol：样本属性个数
Xc = mean(X); % 求原始数据的均值
Xe = std(X); % 求原始数据的标准差
X0 = (X-ones(Xrow,1)*Xc) ./ (ones(Xrow,1)*Xe); % 标准阵X0,标准化为均值0，方差1;
%c = 20000; %此参数可调
for i = 1 : Xrow
for j = 1 : Xrow
K(i,j) = exp(-(norm(X0(i,:) - X0(j,:)))^2/c);%求核矩阵，采用径向基核函数，参数c
end
end
%% 中心化矩阵
n1 = ones(Xrow, Xrow);
N1 = (1/Xrow) * n1;
Kp = K - N1*K - K*N1 + N1*K*N1; % 中心化矩阵
%% 特征值分解
[V, D] = eig(Kp); % 求协方差矩阵的特征向量（V）和特征值（D）
lmda = real(diag(D)); % 将主对角线上为特征值的对角阵变换成特征值列向量
[Yt, index] = sort(lmda, 'descend'); % 特征值按降序排列，t是排列后的数组，index是序号
rate = Yt / sum(Yt); % 计算各特征值的贡献率
sumrate = 0; % 累计贡献率
mpIndex = []; % 记录主元所在特征值向量中的序号
for k = 1 : length(Yt) % 特征值个数
sumrate = sumrate + rate(k); % 计算累计贡献率
mpIndex(k) = index(k); % 保存主元序号
if sumrate > threshold
break;
end
end
npc = length(mpIndex); % 主元个数
%% 计算负荷向量
for i = 1 : npc
zhuyuan_vector(i) = lmda(mpIndex(i)); % 主元向量
P(i,:) = V(:, mpIndex(i)); % 主元所对应的特征向量（负荷向量）
Result(i,:) = X(mpIndex(i),:);
end
zhuyuan_vector2 = diag(zhuyuan_vector); % 构建主元对角阵