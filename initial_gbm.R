# Set up shop
setwd('C:/Users/Mitch/Desktop/Kaggle/Claim Prediction')
source('_utils.R')

# Fit an initial GBM with the following variables:
predictors <- c(
  'ps_car_07_cat', 'ps_ind_06_bin', 'ps_car_02_cat', 'ps_ind_16_bin', 'ps_ind_15', 
  'ps_car_08_cat', 'ps_ind_09_bin', 'ps_car_14', 'ps_reg_03', 'ps_car_03_cat', 
  'ps_car_04_cat', 'ps_ind_07_bin', 'ps_reg_02', 'ps_ind_17_bin', 'ps_car_12', 'ps_car_13'
)

# Read in training data
full <- fread('data/train.csv')

# Random seed for reproducibility
set.seed(731)
full$random <- runif(nrow(full))

train <- full %>% filter(random > 0.3) %>% select(target, predictors)
test <- full %>% filter(random <= 0.3) %>% select(target, predictors)

gbm.fmla <- as.formula(
  paste('~', paste(predictors, collapse = ' + '), '-1')
)

train.mtx <- model.matrix(gbm.fmla, data = train)
test.mtx <- model.matrix(gbm.fmla, data = test)
train.xgb <- xgb.DMatrix(train.mtx, label = train$target, missing = -1)
test.xgb <- xgb.DMatrix(test.mtx, label = test$target, missing = -1)

watchlist <- list(train = train.xgb, test = test.xgb)

gbm1 <- xgb.train(
  eta = 0.02,
  nrounds = 500,
  alpha = 0.1,
  lambda = 0.2,
  data = train.xgb,
  objective = 'binary:logistic',
  eval_metric = .xgbNormalizedGini,
  maximize = TRUE,
  early_stopping_rounds = 10,
  watchlist = watchlist,
  max_depth = 4,
  subsample = 0.9,
  colsample_bytree = 0.9,
  min_child_weight = 50,
  base_score = mean(train$target)
)

importance <- xgb.importance(feature_names = colnames(train.mtx), gbm1)
xgb.ggplot.importance(importance) + theme_fivethirtyeight() + 
  scale_color_fivethirtyeight() + labs(subtitle = 'Initial GBM') + 
  scale_y_continuous(labels = scales::percent)

# idea: fit non-tree model and adjust xgboost predictions by ~5 or 10 points based on 
# predictions of other model
plot_lift(predict(gbm1, test.xgb), test$target, bins = 25, 'Initial GBM')
