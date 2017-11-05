# Set up shop
setwd('C:/Users/Mitch/Desktop/Kaggle/Claim Prediction')
source('_utils.R')

# Read in training data
full <- fread('data/train.csv')

# Names of every predictor
all_predictors <- colnames(full)[!(colnames(full) %in% c('id', 'target'))]

# Random seed for reproducibility
set.seed(731)
full$random <- runif(nrow(full))

# Train/test split
train <- full %>% filter(random > 0.3) %>% select(target, all_predictors)
test <- full %>% filter(random <= 0.3) %>% select(target, all_predictors)

# Formula for GBM
gbm.fmla <- as.formula(
  paste('~', paste(all_predictors, collapse = ' + '), '-1')
)

# Split into matrices
train.mtx <- model.matrix(gbm.fmla, data = train)
test.mtx <- model.matrix(gbm.fmla, data = test)
train.xgb <- xgb.DMatrix(train.mtx, label = train$target, missing = -1)
test.xgb <- xgb.DMatrix(test.mtx, label = test$target, missing = -1)

# Fit the model
watchlist <- list(train = train.xgb, test = test.xgb)
gbm1 <- xgb.train(
  eta = 0.015,
  nrounds = 500,
  data = train.xgb,
  objective = 'binary:logistic',
  eval_metric = .xgbNormalizedGini,
  maximize = TRUE,
  early_stopping_rounds = 10,
  watchlist = watchlist,
  max_depth = 5,
  subsample = 0.95,
  colsample_bytree = 0.95,
  alpha = 0.9,
  lambda = 1.75,
  max_delta_step = 2,
  min_child_weight = 150,
  base_score = mean(train$target)
)

# Check out gain distribution
importance <- xgb.importance(feature_names = colnames(train.mtx), gbm1)
xgb.ggplot.importance(importance[1:25,]) + theme_fivethirtyeight() + 
  scale_color_fivethirtyeight() + labs(subtitle = 'Initial GBM') + 
  scale_y_continuous(labels = scales::percent)

# Plot lift
plot_lift(predict(gbm1, test.xgb), test$target, bins = 25, 'Initial GBM')

# Read in target set
holdout <- fread('data/test.csv')
holdout.mtx <- model.matrix(gbm.fmla, data = holdout %>% select(all_predictors))
holdout.xgb <- xgb.DMatrix(holdout.mtx, missing = -1)

holdout$target <- predict(gbm1, newdata = holdout.xgb)
predictions <- holdout %>% select(id, target)

fwrite(predictions, 'predictions/initial_gbm.csv')

rm(list = ls())
gc()
# idea: fit non-tree model and adjust xgboost predictions by ~5 or 10 points based on 
# predictions of other model (or just average/wtd average predictions)