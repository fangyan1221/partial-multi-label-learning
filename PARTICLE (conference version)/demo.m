%DEMO This is an examplar file on how the PARTICLE program could be used
%The main function is "PAR_train.m" , "PAR_predict.m" ,"PAR_VLS" and "PAR_MAP".
%
%
% Copyright: 
%   Jun-Peng Fang (fangjp@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University,
%   Nanjing 211189, China
%

% Loading the file containing the necessary inputs for calling the PARTICLE function
load('sample data');

% Set parameters for the PARTICLE algorithm
nfold = 10;                %ten fold crossvalidation
k=10;                       % k-nearstneighbor
alpha=0.95;              % A balancing coefficient parameter
str=' -t 0 -c 1';         % libsvm parameter
o=0.9;                      % label confidence threshold parameter
mode=1;                  % 0 means PARTICLE-VLS, 1 means PARTICLE-MAP

[n_sample,~]= size(data);
result=zeros(nfold,5); %save evaluation result
n_test = round(n_sample/nfold);
prelab=[];  %save step 1 result, the most credible labels.
I = 1:n_sample;
for i=1:nfold%
    fprintf('data2 processing,Cross validation: %d\n', i);
    start_ind = (i-1)*n_test + 1;
    if i==nfold
        test_ind = start_ind:n_sample;
    else
        test_ind = start_ind:start_ind+n_test-1;
    end
    train_ind = setdiff(I,test_ind);
    train_data = data(train_ind, :);
    train_p_target = partial_labels(:,train_ind);
    test_data = data(test_ind,:);
    test_p_target = partial_labels(:, test_ind);
    model = PAR_train(train_data,train_p_target,k,alpha);
    lab = PAR_predict(train_data,test_data,test_p_target,model,o);
    prelab=[prelab,lab];
end

for i=1:nfold%
    fprintf('result processing,Cross validation: %d\n', i);
    start_ind = (i-1)*n_test + 1;
    if i==nfold
        test_ind = start_ind:n_sample;
    else
        test_ind = start_ind:start_ind+n_test-1;
    end
    train_ind = setdiff(I,test_ind);
    train_data = data(train_ind, :);
    train_target = prelab(:,train_ind);
    pre_target=partial_labels(:,train_ind);
    test_data = data(test_ind,:);
    test_target = target(:, test_ind);
    pre_test_target = prelab(:, test_ind);
    if mode==0
        [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision] = PAR_VLS( train_data,train_target,pre_target,test_data,test_target,str);
    elseif mode==1
        [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision] = PAR_MAP( train_data,train_target,pre_target,test_data,test_target,str);
    end
    result(i,:) = [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision];
end
rr=sum(result)/nfold;

