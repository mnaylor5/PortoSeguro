# Set up shop
setwd('C:/Users/Mitch/Desktop/Kaggle/Claim Prediction')
source('_utils.R')

# Fit an initial GBM with the following variables:
# Likely still a few predictive features being left out in the correlation_eda bit
predictors <- c(
  'ps_car_07_cat', 'ps_ind_06_bin', 'ps_car_02_cat', 'ps_ind_16_bin', 'ps_ind_15', 
  'ps_car_08_cat', 'ps_ind_09_bin', 'ps_car_14', 'ps_calc_19_bin', 'ps_car_11', 
  'ps_car_15', 'ps_ind_05_cat', 'ps_reg_03', 'ps_car_03_cat', 'ps_car_04_cat', 
  'ps_ind_07_bin', 'ps_reg_02', 'ps_ind_17_bin', 'ps_car_12', 'ps_car_13'
)

# Read in training data
full <- fread('train.csv')

# Random seed for reproducibility
set.seed(731)
full$random <- runif(nrow(full))

# Train/test split
train <- full %>% filter(random > 0.3) %>% select(target, predictors)
test <- full %>% filter(random <= 0.3) %>% select(target, predictors)

# Formula for GBM
gbm.fmla <- as.formula(
  paste('~', paste(predictors, collapse = ' + '), '-1')
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
  alpha = 0.1,
  lambda = 1.5,
  max_delta_step = 2,
  min_child_weight = 50,
  base_score = mean(train$target)
)

# Check out important predictors
importance <- xgb.importance(feature_names = colnames(train.mtx), gbm1)
xgb.ggplot.importance(importance) + theme_fivethirtyeight() + 
  scale_color_fivethirtyeight() + labs(subtitle = 'Initial GBM') + 
  scale_y_continuous(labels = scales::percent)

# Plot lift
plot_lift(predict(gbm1, test.xgb), test$target, bins = 25, 'Initial GBM')

# idea: fit non-tree model and adjust xgboost predictions by ~5 or 10 points based on 
# predictions of other model (or just average/wtd average predictions)