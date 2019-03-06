function DiceCoefficient=Calculate_DiceCoefficient(GroundTruth,Predicted)
%% This code is written for Image Segmentation (EM algorithm)
% This code will return the dice- coefficient for two Images.
% Written By Md. Kamrul Hasan, 
%contact: kamruleeekuet@gmail.com
% GroundTruth Means GT Image
% Predicted Means Predicted Image
%%
TruePositive=0;
TrueNegative=0;
FalsePositive=0;
FalseNegative=0;

[row_GT,col_GT]=size(GroundTruth);
[row_Predicted,col_Predicted]=size(Predicted);

if row_GT==row_Predicted && col_GT==col_Predicted
    for i=1:1:row_Predicted
        for j=1:1:col_GT
            if GroundTruth(i,j)==0
                if GroundTruth(i,j)==Predicted(i,j)
                    TrueNegative=TrueNegative+1;
                else
                    FalsePositive=FalsePositive+1;
                end          
            else
                if GroundTruth(i,j)==Predicted(i,j)
                    TruePositive=TruePositive+1;
                else
                    FalseNegative=FalseNegative+1;
                end         
            end
        end
    end
    DiceCoefficient=((2*TruePositive)/(2*TruePositive+FalsePositive+FalseNegative));
else
    disp('Sorry!! You are supposed to have same dimentional Image to Calculate DSC.');
end
%% ---------------------------- The End--------------------------------------