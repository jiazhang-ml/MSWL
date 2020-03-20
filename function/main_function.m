
function [W, obj] = main_function( X_train, Y_train, para, S, B, p )

[~, dim] = size(X_train); 

[~, label_num] = size(Y_train);
if para.m==para.n  
    U1([1:para.m],1) = 1; U=diag(U1);
else 
    U1([1:para.m],1) = 1; U2([1:para.n-para.m],1) = p; U = [U1;U2]; U=diag(U);
end
clear U1 U2;

%Initialize W
W = rand(dim, label_num); L = speye(label_num);

iter = 1; obji = 1; 

while 1
 
    % update W
    d = 0.5./sqrt(sum(W.*W, 2) + eps); 
    D = diag(d);
    
    GG1 = (X_train - S * X_train)' * (X_train - S * X_train);
    [r1,r2]=find(isnan(GG1)); GG1(r1,r2)=0;
    
    A = X_train' * U * X_train + para.lamda1 * GG1 + para.lamda3 * D;
    [r3,r4]=find(A==inf); A(r3,r4)=0;
    
    C = para.lamda2 * (eye(label_num)-B)' * (eye(label_num)-B); % B'~=B
    E = -X_train' * U * Y_train;
    W = lyap(A, C, E); 
    
    F = X_train * W;
    
    for i = para.m+1:para.n
        for j = 1:label_num
            if F(i, j) <= 0
                Y_train(i, j) = 0;
            else
                if F(i, j) >= 1
                    Y_train(i, j) = 1;
                else
                    Y_train(i, j) = F(i ,j);
                end
            end
        end
    end

%     L = ((S * Y_train)' * U * S * Y_train) \ ((S * Y_train)' * U * X_train * W);
    
    GG2=(norm((X_train * W - S * X_train * W), 'fro'))^2;
    [r5,r6]=find(isnan(GG2)); GG2(r5,r6)=0;
    
    obj(iter) = (norm((X_train * W - Y_train), 'fro'))^2 ...
        + para.lamda1 * GG2 + para.lamda2 * (norm((W - W * B), 'fro'))^2 + para.lamda3 * sum(sqrt(sum(W.*W,2) + eps));

    cver = abs((obj(iter) - obji)/obji);
    obji = obj(iter); 
    iter = iter + 1;
    if (iter == 20) , break, end
    if (cver < 10^-3 && iter > 2) , break, end
end

end

