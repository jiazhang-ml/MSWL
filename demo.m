% This is an example file on how the MSWL program could be used.

% Please feel free to contact me (zhangjia_gl@163.com), if you have any problem about this program.

clear;clc;
addpath(genpath('.\'))
data_name='stackexchess'; load stackexchess.mat

para.lamda1 = 0.1;   % lamda1_searchrange = 10.^[-3:3]; 
para.lamda2 = 0.1;  % lamda2_searchrange = 10.^[-3:3];
para.lamda3 = 1;     % lamda3_searchrange = 10.^[-6:6];

para.mu     = 0.1;
p           = 0.5;
cvx_setup

para.rep    = 5;

incomplete  = 0.3; %randomly dropping ~% of the observed labels

if exist('X_train','var')==1
    data=[X_train; X_test];
    target=[Y_train; Y_test];
    clear X_train X_test Y_train Y_test
end
ind=find(target==-1); target(ind)=0;
num_data = size(data,1);
    
randorder = randperm(num_data);
PRO=zeros(4,para.rep);

for t = 1:para.rep
	
	[X_train, Y_train, X_test, Y_test] = generateCVSet( data,target,randorder,t,para.rep );
        
    [train_num,~] = size(X_train);
    labeled_proportion = 1; % labeled and unlabeled data
            
    
    [~,label_dim] = size(Y_train);

    Y_train(Y_train == 0) = -1; Y_test(Y_test == 0) = -1;

    R = randperm(train_num);

    labeled_num = round(train_num*labeled_proportion);
    unlabel_num = train_num-labeled_num;

    random_train_Index = randperm(labeled_num*label_dim);

    Train_Matrix = X_train(R(1:labeled_num),:);
    Train_Label = Y_train(R(1:labeled_num),:);
    if unlabel_num==0
      UnLabel_Matrix=[];
    else
      UnLabel_Matrix = X_train(R(labeled_num+1:labeled_num + unlabel_num),:);
    end
    Test_Matrix = X_test;
    Test_Label = Y_test;

    Z = createIncomplete_Label(Train_Label, incomplete,random_train_Index);
 
    X_train = [Train_Matrix; UnLabel_Matrix];

    [para.m,~] = size(Train_Matrix); [para.n,~] = size(X_train);     
            
    X_test = Test_Matrix;

    [~, label_num] = size(Y_train); Y_train = [Z;zeros(para.n-para.m,label_num)]; 

    Y_test = Test_Label;

    Y_train(Y_train == -1) = 0; 

    K = label_num + 1; S = estimate_top_struct(X_train, K);
    for i = 1:para.n
        S(i,i) = 1; 
    end
    B = [];
    for i = 1:label_num
        cvx_begin
        variable V2(label_num);
        minimize(norm(Y_train(:,i) - Y_train*V2) + para.mu * norm(V2,1));
        subject to
            V2(i)==0;
        cvx_end
        B=[B V2];    
    end
    
%     B = rec_label(Y_train);
    for i = 1:label_num
        B(i,i) = 1; % keep original labels
    end
    
    Y_train = Y_train * B;
    P = find(Y_train > 1); Y_train(P) = 1;
    Q = find(Y_train < 0); Y_train(Q) = 0;
    
    %run main function
    tic;
    [W, obj] = main_function( X_train, Y_train, para, S, B, p);
    tm = toc;
    % predict
    Outputs = X_test*W;
    
    TT(:,1) = EvaluationAll(Outputs',Y_test');
            
    Y_test(Y_test == -1) = 0; 
    [auc,~,~] = ak_auc_tp_fp_diffrent_ks(Outputs,Y_test); TT(1,1)=auc;

    PRO(:,t) = TT;
end
Avg_Result_OURS_3(:,1)=mean(PRO,2); Avg_Result_OURS_3(:,2)=std(PRO,1,2);