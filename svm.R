# Fit SVM with all predictors 
setwd('C:/Users/Mitch/Desktop/Kaggle/Claim Prediction')
source('_utils.R')
library(liquidSVM)

# Read in training data
full <- fread('data/train.csv')

# Names of every predictor
all_predictors <- colnames(full)[!(colnames(full) %in% c('id', 'target'))]

# Random seed for reproducibility
# Different seed from GBM
set.seed(865)
full$random <- runif(nrow(full))

# Train/test split
train <- full %>% filter(random > 0.3) %>% 
  mutate(target = as.factor(target)) %>%
  select(target, all_predictors)

test <- full %>% filter(random <= 0.3) %>% 
  mutate(target = as.factor(target)) %>%
  select(target, all_predictors)

# Formula
svm.fmla <- as.formula(
  paste('~', paste(all_predictors, collapse = ' + '), '-1')
)

# Fit SVM
start <- Sys.time() # kick off a timer
svm_test <- svm(target ~ ., 
                train, 
                testdata = test %>% select(-target), 
                testdata_labels = test$target, 
                predict.prob = T,
                max_gamma = 25)
Sys.time() - start

prediction <- predict(svm_test, test %>% select(-target))
colnames(prediction) <- c('p0', 'p1')
summary(test$prediction)
str(test$prediction)
