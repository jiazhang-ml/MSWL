
function ResultAll = EvaluationAll(Outputs,test_target)
% evluation for MLC algorithms, there are fifteen evaluation metrics
% 
% syntax
%   ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)

    
    ResultAll = zeros(4,1); 

    RankingLoss       = Ranking_loss(Outputs,test_target);
    Coverage          = coverage(Outputs,test_target);
    Average_Precision = Average_precision(Outputs,test_target);
    
 
    ResultAll(2,1)    = RankingLoss; 
    ResultAll(3,1)    = Coverage; 
    ResultAll(4,1)    = Average_Precision; 
    
 
end