# Load necessary libraries
library(MASS)
library(quantregForest)
library(mboost)
library(caret)
library(foreach)
library(doParallel)
library(tidyr)
library(qrnn)

# Load the Boston dataset
data(Boston)

# Set parameters
quantiles <- c(1:9/10) # Quantiles to evaluate
n_bootstrap <- 6 # Number of bootstrap samples

# Custom function to calculate quantile loss
quantile_loss <- function(actual, predicted, q) {
  error <- actual - predicted
  mean(ifelse(error > 0, q * error, (1 - q) * -error))
}

# Register parallel backend
num_cores <- detectCores() - 1 # Use one less than the available cores
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Parallelized Monte Carlo simulation
results <- foreach(b = 1:n_bootstrap, .combine = rbind, .packages = c("MASS", "mboost", "quantregForest", "caret", "qrnn")) %dopar% {
  # Split data into training and test sets
  train_indices <- sample(1:nrow(Boston), size = 0.7 * nrow(Boston))
  train_data <- Boston[train_indices, ]
  test_data <- Boston[-train_indices, ]
  
  # Placeholder for bootstrap results
  bootstrap_res <- data.frame()
  
  ## --- quantregForest (tuned once) ---
  
  qrf_grid <- expand.grid(mtry = seq(2, ncol(Boston) - 1, by = 2), ntree = seq(100, 500, by = 100))
  qrf_losses <- c()
  
  folds <- createFolds(train_data$medv, k = 5)
  for (i in seq_len(nrow(qrf_grid))) {
    mtry <- qrf_grid$mtry[i]
    ntree <- qrf_grid$ntree[i]
    cv_losses <- c()
    
    for (fold in folds) {
      fold_train <- train_data[-fold, ]
      fold_val <- train_data[fold, ]
      
      model_qrf <- quantregForest(x = fold_train[, -ncol(Boston)], y = fold_train$medv, mtry = mtry, ntree = ntree)
      predictions <- predict(model_qrf, newdata = fold_val[, -ncol(Boston)], what = 0.5) # Using median (tau = 0.5)
      loss <- quantile_loss(fold_val$medv, predictions, 0.5)
      cv_losses <- c(cv_losses, loss)
    }
    qrf_losses <- c(qrf_losses, mean(cv_losses))
  }
  best_qrf_params <- qrf_grid[which.min(qrf_losses), ]
  final_qrf <- quantregForest(x = train_data[, -ncol(Boston)], y = train_data$medv, mtry = best_qrf_params$mtry, ntree = best_qrf_params$ntree)
  
  # --- Default QRF Model ---
  default_qrf <- quantregForest(x = train_data[, -ncol(Boston)], y = train_data$medv)
  
  for (q in quantiles) {
    
    # --- Hyperparameter Tuning GamBoost for each quantile ---
    
    gamboost_grid <- expand.grid(mstop = seq(100, 500, by = 100))
    gamboost_losses <- c()
    
    for (mstop in gamboost_grid$mstop) {
      cv_losses <- c()
      for (fold in folds) {
        fold_train <- train_data[-fold, ]
        fold_val <- train_data[fold, ]
        
        model_gamboost <- gamboost(medv ~ ., data = fold_train, family = QuantReg(tau = q), control = boost_control(mstop = mstop), baselearner = "bols")
        predictions <- predict(model_gamboost, newdata = fold_val)
        loss <- quantile_loss(fold_val$medv, predictions, q)
        cv_losses <- c(cv_losses, loss)
      }
      gamboost_losses <- c(gamboost_losses, mean(cv_losses))
    }
    best_mstop <- gamboost_grid$mstop[which.min(gamboost_losses)]
    
    # --- Final tuned gamboost model
    final_gamboost <- gamboost(medv ~ ., data = train_data, family = QuantReg(tau = q), control = boost_control(mstop = best_mstop), baselearner = "bols")
    gamboost_predictions <- predict(final_gamboost, newdata = test_data)
    gamboost_test_loss <- quantile_loss(test_data$medv, gamboost_predictions, q)
    bootstrap_res <- rbind(bootstrap_res, data.frame(Bootstrap = b, Quantile = q, Method = "gamboost (tuned)", Loss = gamboost_test_loss))
    
    # --- Default gamboost
    default_gamboost <- gamboost(medv ~ ., data = train_data, family = QuantReg(tau = q), baselearner = "bols")
    default_gamboost_predictions <- predict(default_gamboost, newdata = test_data)
    default_gamboost_test_loss <- quantile_loss(test_data$medv, default_gamboost_predictions, q)
    bootstrap_res <- rbind(bootstrap_res, data.frame(Bootstrap = b, Quantile = q, Method = "gamboost (default)", Loss = default_gamboost_test_loss))
    
    # --- Tuned quantregForest
    qrf_predictions <- predict(final_qrf, newdata = test_data[, -ncol(Boston)], what = q)
    qrf_test_loss <- quantile_loss(test_data$medv, qrf_predictions, q)
    bootstrap_res <- rbind(bootstrap_res, data.frame(Bootstrap = b, Quantile = q, Method = "quantregForest (tuned)", Loss = qrf_test_loss))
    
    # -- Default quantregForest
    default_qrf_predictions <- predict(default_qrf, newdata = test_data[, -ncol(Boston)], what = q)
    default_qrf_test_loss <- quantile_loss(test_data$medv, default_qrf_predictions, q)
    bootstrap_res <- rbind(bootstrap_res, data.frame(Bootstrap = b, Quantile = q, Method = "quantregForest (default)", Loss = default_qrf_test_loss))
  }
  
  bootstrap_res
}

# Stop parallel backend
stopCluster(cl)

# Combine all results into a single data frame
final_results <- results

# Calculate average quantile loss for each method and quantile
summary_results <- aggregate(Loss ~ Quantile + Method, data = final_results, FUN = mean)

# Display the results
print(summary_results)

# Pivot wider to make quantiles the columns
formatted_results <- summary_results %>%
  pivot_wider(names_from = Quantile, values_from = Loss)

# Rename columns for better readability
colnames(formatted_results) <- c("Method", paste0("Quantile_", quantiles))

# Display the formatted results
print(formatted_results)
