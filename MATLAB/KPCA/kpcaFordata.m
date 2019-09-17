function [P,mpIndex,Result] = kpcaFordata(X,c,threshold) %
%load tsstdata  %�����������
%X = testdata(:,1:4)'; % Xѵ�����ݼ������ݵ���Ϊ������������Ϊһ������
%%�����ݹ�һ������һ��Ϊ��ֵΪ0������Ϊ1
[Xrow, Xcol] = size(X); % Xrow���������� Xcol���������Ը���
Xc = mean(X); % ��ԭʼ���ݵľ�ֵ
Xe = std(X); % ��ԭʼ���ݵı�׼��
X0 = (X-ones(Xrow,1)*Xc) ./ (ones(Xrow,1)*Xe); % ��׼��X0,��׼��Ϊ��ֵ0������1;
%c = 20000; %�˲����ɵ�
for i = 1 : Xrow
for j = 1 : Xrow
K(i,j) = exp(-(norm(X0(i,:) - X0(j,:)))^2/c);%��˾��󣬲��þ�����˺���������c
end
end
%% ���Ļ�����
n1 = ones(Xrow, Xrow);
N1 = (1/Xrow) * n1;
Kp = K - N1*K - K*N1 + N1*K*N1; % ���Ļ�����
%% ����ֵ�ֽ�
[V, D] = eig(Kp); % ��Э������������������V��������ֵ��D��
lmda = real(diag(D)); % �����Խ�����Ϊ����ֵ�ĶԽ���任������ֵ������
[Yt, index] = sort(lmda, 'descend'); % ����ֵ���������У�t�����к�����飬index�����
rate = Yt / sum(Yt); % ���������ֵ�Ĺ�����
sumrate = 0; % �ۼƹ�����
mpIndex = []; % ��¼��Ԫ��������ֵ�����е����
for k = 1 : length(Yt) % ����ֵ����
sumrate = sumrate + rate(k); % �����ۼƹ�����
mpIndex(k) = index(k); % ������Ԫ���
if sumrate > threshold
break;
end
end
npc = length(mpIndex); % ��Ԫ����
%% ���㸺������
for i = 1 : npc
zhuyuan_vector(i) = lmda(mpIndex(i)); % ��Ԫ����
P(i,:) = V(:, mpIndex(i)); % ��Ԫ����Ӧ����������������������
Result(i,:) = X(mpIndex(i),:);
end
zhuyuan_vector2 = diag(zhuyuan_vector); % ������Ԫ�Խ���