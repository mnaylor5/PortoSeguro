# Utility functions for Porto Seguro Kaggle competition
library(ggplot2)
library(ggthemes)
library(scales)
library(xgboost)
library(dplyr)
library(data.table)

# Plot lift
# Args
#   - Actuals: Numeric vector of actual
#   - Predicted: Numeric vector of predictions
#   - Bins: Predictions are ordered and bucketed into n even buckets
#   - Subtitle: String that appears as the subtitle (under "Modeled Lift")
# Output
#   - ggplot object with lines for predicted and actual
plot_lift <- function(predicted, actual, bins = 25, subtitle = NULL){
  df <- data.frame(preds = predicted, act = actual) %>%
    arrange(preds) %>% mutate(bucket = cut_number(preds, bins)) %>%
    group_by(bucket) %>% summarise(Predicted = mean(preds), Actual = mean(act)) %>%
    melt(id='bucket')
  
  ggplot(df, aes(x = as.numeric(bucket) * 100/bins, y = value, 
                 col = variable, fill = variable, group = variable)) + 
    geom_line() + geom_point() + theme_bw() + 
    labs(title = 'Modeled Lift', y = 'Average Rate of Claim Filing', 
         x = 'Prediction Percentile', col = NULL, fill = NULL, subtitle = subtitle) + 
    scale_y_continuous(labels = scales::percent)
}

# Normalized Gini coefficient
# Args:
#   - Predicted
#   - Actual
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) 
    accum.losses <- temp.df$actual / total.losses
    gini.sum <- cumsum(accum.losses - null.losses)
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

# Normalized Gini coefficient as an XGBoost cost function
.xgbNormalizedGini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, 'label')
  return(list(metric = 'NormGini', value = normalizedGini(labels, preds)))
}
