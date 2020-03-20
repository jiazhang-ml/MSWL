Y_test  = vec_read('corel5k_test_annot.hvecs');
X_test  = vec_read('corel5k_test_DenseSift.hvecs');
Y_train = vec_read('corel5k_train_annot.hvecs');
X_train = vec_read('corel5k_train_DenseSift.hvecs');

Y_test  = double(Y_test);
X_test  = double(X_test);
Y_train = double(Y_train);
X_train = double(X_train);

%%%%%%%%%%%%%%%%%%%%%L2·¶Êı¹éÒ»»¯%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist('X_train','var')==1
    data    = [X_train;X_test]; target  = [Y_train;Y_test];
end
data        = double (data);
num_data    = size(data,1);
temp_data   = data + eps;

temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
if sum(sum(isnan(temp_data)))>0
    temp_data = data+eps;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
end

clear data;
data = temp_data;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
