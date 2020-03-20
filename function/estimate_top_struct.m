
function S = estimate_top_struct(X, K)
% 
% ESTIMATE_TOP_STRUCT      Estimate the topological structure in the feature space.
% 
% Inputs:
%   X: data matrix with training samples in rows and features in columns (N x D)
%   K: number of selected nearest neighbors.
%     
% Output:
% 	S: weight matrix

fprintf(1,'Estimate the topological structure.\n');

[N,D] = size(X);

neighborhood = knnsearch(X, X, 'K', K+1);
neighborhood = neighborhood(:, 2:end);

if(K>D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

% Least square programming
S = sparse(N, N);
for i=1:N
    neighbors = neighborhood(i,:);
    z = X(neighbors,:)-repmat(X(i,:),K,1); % shift ith pt to origin
    Z = z*z';                                        % local covariance
    Z = Z + eye(K,K)*tol*trace(Z);                   % regularlization (K>D)
    S(i,neighbors) = Z\ones(K,1);                           % solve Zw=1
    S(i,neighbors) = S(i,neighbors)/sum(S(i,neighbors));                  % enforce sum(w)=1
end
S=full(S); [m,n]=find(isnan(S)); S(m,n)=0;
end