# Kaggle Claim Prediction
## Overview
Brazilian insurance company Porto Seguro would like to predict which policies are more 
likely to file a claim within a year. The evaluation metric is normalized Gini coefficient,
which measures how well predictions segment the target.

## Data
The training dataset consists of around 595,000 rows and 59 columns, including the dependent 
variable `target`. The features in the training set have been given nondescript names, such 
as `ps_ind_01`, so any attempt to relate customer information to predictor variables will be
met with considerable difficulty.

## Modeling Approaches
My initial approach is an `xgboost` model -- I use a fair amount of L1 and L2 regression to 
avoid overfitting and improve test performance.