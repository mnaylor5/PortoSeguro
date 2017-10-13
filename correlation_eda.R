# Playing around with the Porto Seguro claims data
setwd('C:/Users/Mitch/Desktop/Kaggle/Claim Prediction')
source('_utils.R')

# Read in training data (~595k obs, 59 col)
train <- fread('train.csv')

# Check out the target variable 
# ~3.6% of policyholders file a claim within a year on average
summary(train$target)

# Correlation matrix
cor_mtx <- as.data.frame(cor(train %>% select(-id))) %>%
  mutate(feature = rownames(.), correlation = scales::percent(target)) %>% 
  arrange(target) %>% select(feature, correlation) %>% filter(feature != 'target') 

head(cor_mtx); tail(cor_mtx)

# Trying the top/bottom 5 correlation plots
top_vars <- rbind(head(cor_mtx, 10), tail(cor_mtx, 10))
one_way_avg <- list()

for(i in 1:nrow(top_vars)){
  name <- top_vars$feature[i]
  predictor <- as.vector(train[[name]])
  oneWay <- tapply(train$target, predictor, mean)
  ggsave(
    paste0('figure/One-Way EDA/', name, '.png'),
    ggplot(train, aes(x = as.factor(predictor), y = target, group = 1)) + 
      geom_smooth(se = F, stat = 'summary', fun.y = mean) + 
      theme_fivethirtyeight() + scale_color_fivethirtyeight() + 
      labs(title = 'One-Way Plots', subtitle = name)
  )
}

tapply(train$target, train$ps_car_07_cat, length) # 1.9% missing
tapply(train$target, train$ps_ind_06_bin, length) # none missing, highly correlated
tapply(train$target, train$ps_car_02_cat, mean) # essentially no missing values, high correlation
tapply(train$target, train$ps_ind_16_bin, mean) 
tapply(train$target, train$ps_ind_15, mean)
