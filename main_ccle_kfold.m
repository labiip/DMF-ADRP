clear all;close all;clc;

% import matrix for completion

Y_filepath = 'ccle.xlsx';
[Y, text, alldata] = xlsread(Y_filepath);

Y = Y'; % 491*23 --to--23*491

Y_dim = size(Y);

% Known_index0 = find(isnan(Y)==0);
% Missing_index0 = find(isnan(Y)==1);
% missing_index_1 = randperm(length(Known_index0),500);
% Known_index = setdiff(Known_index0,Known_index0(missing_index_1));
% Missing_index = union(Missing_index0,Known_index0(missing_index_1));

Known_index = find(isnan(Y)==0);
Missing_index = find(isnan(Y)==1);

% data global effect preprocessing
global_mean = nanmean(Y(:));
Y1G = Y - global_mean;
column_mean = nanmean(Y1G,1);
Y2C = Y1G - column_mean;
row_mean = nanmean(Y2C,2);
Y3R = Y2C - row_mean; 
Y3R(Missing_index) = 0;

% replace NaN with 0 in Y
Y_0 = Y;
Y_0(Missing_index) = 0;

Mask_mat = ones(size(Y));
Mask_mat(Missing_index) = 0;
Mask_mat = logical(Mask_mat);

Y_recover1 = zeros(Y_dim);
result = zeros(30,9);
for cv_run = 1:1
    disp(sprintf('Run %d:',cv_run));
    rand('seed',cv_run + 20000);
    K_fold = 10;
    indices = crossvalind('Kfold',Known_index,K_fold);

for fold_id = 1:10
    % cross valind for Y
    disp(sprintf('Fold %d:',fold_id));
    test = (indices == fold_id); train = ~test;
%     Y_tmp = Y_0;% complete data (original)
    Y_tmp = Y3R;% global effect data (original)
    Y_tmp(Known_index(test)) = 0;% incomplete data masked by k-fold
%     max_Y_tmp = max(max(Y_tmp));
%     mean_Y_tmp = (max(Y_tmp(Known_index(train)))-min(Y_tmp(Known_index(train))))/2;
%     Y_tmp = Y_tmp - mean_Y_tmp;
    % Y_tmp = Y_tmp / max_Y_tmp;
    Mask_test = Mask_mat;
    Mask_test(Known_index(test)) = 0;
    M = (Mask_test - Mask_mat)+1;
    
    m=23;
    n=491;
    r=40;
    
    % DMF setup
    s=[r 160 m];% input size, hidden size 1, ..., output size
    options.Wp=0.01;
    options.Zp=0.01;
    options.maxiter=3500;
    % 'tanh_opt','sigm','linear'
    options.activation_func={'tanh_opt','linear'};
    [Y_DMF,NN_MF]=MC_DMF(Y_tmp',Mask_test',s,options);
    Yr=Y_DMF';
    Yr = Yr + row_mean + column_mean + global_mean;
    % compute recovery error
    re_error=norm((Y_0-Yr).*(1-M),'fro')/norm(Y_0.*(1-M),'fro');
    
    disp(['Relative recovery error is ' num2str(re_error)]) ;
    Y_recover1(Known_index(test)) = Yr(Known_index(test));
end
end

    Y_0 = Y_0';
    Y_recover1 = Y_recover1';
    
    figure;
    subplot(2,2,1);imagesc(Y_0);colorbar;
    subplot(2,2,2);imagesc(Y_recover1);colorbar;
    subplot(2,2,3);scatter(Y_0(Known_index),Y_recover1(Known_index),'.');%corr(Y_0(Known_index),Y_recover1(Known_index))
%     subplot(2,2,4);plot(1:length(obj_rec),obj_rec);
    % scatter(Y_0(Known_index),Y_recover2(Known_index),'.');%corr(Y_0(Known_index),Y_recover2(Known_index))
    global_pcc = corr(Y_0(Known_index),Y_recover1(Known_index));
    num = Y_0';
    numpred = Y_recover1';

    drugwisecorr = NaN(size(num,1),1);
    drugwise_qt = NaN(size(num,1),1);
    drugwiseerr = NaN(size(num,1),1);
    drugwiseerr_qt = NaN(size(num,1),1);
    drugwiserepn = NaN(size(num,1),1);
for d = 1:size(num,1)
    curtemp1 = num(d,:);
    y1 = prctile(curtemp1,75);
    xia1 = find(curtemp1 >= y1);
    y2 = prctile(curtemp1,25);
    xia2 = find(curtemp1 <= y2);
    xia = [xia1,xia2];
    drugwise_qt(d) = corr(curtemp1(xia)',numpred(d,xia)');
    drugwiseerr_qt(d) = sqrt(sum((curtemp1(xia)-numpred(d,xia)).^2)/sum(~isnan(curtemp1(xia))));
    curtemp2 = numpred(d,:);
    curtemp2(isnan(curtemp1)) = [];
    curtemp1(isnan(curtemp1)) = [];
    drugwiserepn(d) = length(curtemp1);  
    drugwisecorr(d) = corr(curtemp1',curtemp2');
    drugwiseerr(d) = sqrt(sum((curtemp1-curtemp2).^2)/sum(~isnan(curtemp1)));
end
    ave_pcc = mean(drugwisecorr);
    std_pcc = std(drugwisecorr);
    ave_err = mean(drugwiseerr);
    std_err = std(drugwiseerr);
    ave_pcc_sr = mean(drugwise_qt);
    std_pcc_sr = std(drugwise_qt);
    ave_err_sr = mean(drugwiseerr_qt);
    std_err_sr = std(drugwiseerr_qt);
    disp(sprintf('ave_err:%d,ave_err_sr:%d,ave_pcc:%d,ave_pcc_sr:%d',ave_err,ave_err_sr,ave_pcc,ave_pcc_sr));
    result(cv_run, :) = [global_pcc ave_pcc ave_pcc_sr ave_err ave_err_sr std_pcc std_pcc_sr std_err std_err_sr];
    
