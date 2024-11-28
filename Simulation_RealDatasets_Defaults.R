library(MASS)
library(quantreg)
library(quantregForest)
library(mboost)
library(caret)
library(foreach)
library(doParallel)
library(qrnn)
library(dplyr)
library(gamlss)
library(gamboostLSS)
library(scoringRules)
library(Rearrangement)

# Boston Dataset
data(Boston)
head(Boston)
Boston_new <- Boston[, c("medv", setdiff(names(Boston), "medv"))]
colnames(Boston_new)[1] <- "y"
head(Boston_new)

# Mtcars Dataset
data(mtcars)
head(mtcars)
mtcars_new <- mtcars
colnames(mtcars_new)[1] <- "y"
head(mtcars_new)

# Swiss Dataset
data(swiss)
head(swiss)
swiss_new <- swiss
colnames(swiss_new)[1] <- "y"
head(mtcars_new)

simulation_function <- function(data, quantiles, n_bootstrap, seed) {
  
  crps_quantiles <- function(actual, predictions, q, method) {
    
    methods_with_rearrangement <- c("mboost", "lightgbm")
    
    n <- length(actual)
    M <- length(quantiles)
    
    if (method %in% methods_with_rearrangement) {
      rearranged_predictions <- t(apply(predictions, 1, function(pred) {
        rearrangement(as.data.frame(quantiles), pred)
      }))
    } else {
      rearranged_predictions <- predictions
    }
    
    quantile_loss <- function(y, q_value, q) {
      error <- y - q_value
      (q - ifelse(error < 0, 1, 0)) * error
    }
    
    crps_per_observation <- sapply(1:n, function(i) {
      losses <- sapply(1:M, function(m) {
        quantile_loss(actual[i], rearranged_predictions[i, m], quantiles[m])
      })
      (2 / M) * sum(losses)
    })
    
    mean_crps <- mean(crps_per_observation)
  }
  
  # Register parallel backend
  num_cores <- detectCores() - 1 # Use one less than the available cores
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  # Parallelized Monte Carlo simulation
  results <- foreach(b = 1:n_bootstrap, .combine = rbind, .packages = c("MASS", 
              "mboost", "quantregForest", "caret", "quantreg", "lightgbm", "qrnn", 
              "dplyr", "gamlss", "gamboostLSS", "scoringRules", "Rearrangement")) %dopar% {
    
    set.seed(seed + b)
    # Split data into training and test sets
    train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
    
    # Placeholder for CRPS results
    crps_res <- data.frame()
    
    gamlls_normal <- mboostLSS(y ~ .,
                     data = train_data, 
                     families = GaussianLSS(), baselearner = "bols")
    params_test_normal <- predict(gamlls_normal, newdata = test_data)
    gamlss_normal_crps <- mean(crps_norm(y = test_data$y, location = params_test_normal$mu, scale = params_test_normal$sigma))
    crps_res <- rbind(crps_res, data.frame(Bootstrap = b, Method = "GAMLSSBOOST Normal", CRPS = gamlss_normal_crps))
    
    ## --- quantregForest ---
    default_qrf <- quantregForest(x = train_data[, -1], y = train_data$y)
    qrf_predictions <- predict(default_qrf, newdata = test_data[, -1], what = quantiles)
    qrf_crps <- crps_quantiles(test_data$y, qrf_predictions, quantiles, method = "qrf")
    crps_res <- rbind(crps_res, data.frame(Bootstrap = b, Method = "quantregForest", CRPS = qrf_crps))
    
    ## --- LASSO ---
    default_lasso <- rq(y ~ ., tau = quantiles, data = train_data, method = "lasso")
    default_lasso_predictions <- predict(default_lasso, newdata = test_data, tau = quantiles)
    lasso_crps <- crps_quantiles(test_data$y, default_lasso_predictions, quantiles, method = "lasso")
    crps_res <- rbind(crps_res, data.frame(Bootstrap = b, Method = "LASSO", CRPS = lasso_crps))
    
    ## --- GamBoost ---
    gamboost_predictions <- sapply(quantiles, function(q) {
      gamboost_model <- gamboost(y ~ ., data = train_data, family = QuantReg(tau = q), baselearner = "bols")
      predict(gamboost_model, newdata = test_data[,-1])
    })
    gamboost_crps <- crps_quantiles(test_data$y, gamboost_predictions, quantiles, method = "mboost")
    crps_res <- rbind(crps_res, data.frame(Bootstrap = b, Method = "GamBoost", CRPS = gamboost_crps))
    
    ## ---blackboost - 
    blackboost_predictions <- sapply(quantiles, function(q) {
      blackboost_model <- blackboost(y ~ ., data = train_data, family = QuantReg(tau = q))
      predict(blackboost_model, newdata = test_data[,-1])
    })
    blackboost_crps <- crps_quantiles(test_data$y, blackboost_predictions, quantiles, "mboost")
    crps_res <- rbind(crps_res, data.frame(Bootstrap = b, Method = "BlackBoost", CRPS = blackboost_crps))
    
    ## --- LightGBM ---
    lgb_predictions <- sapply(quantiles, function(q) {
      lgb_train <- lgb.Dataset(data = as.matrix(train_data[, -1]), label = train_data$y)
      lgb_params <- list(objective = "quantile", metric = "quantile", alpha = q)
      lgb_model <- lgb.train(params = lgb_params, data = lgb_train, verbose = -1)
      predict(lgb_model, as.matrix(test_data[, -1]))
    })
    lgb_crps <- crps_quantiles(test_data$y, lgb_predictions, quantiles, method = "lightgbm")
    crps_res <- rbind(crps_res, data.frame(Bootstrap = b, Method = "LightGBM", CRPS = lgb_crps))
    
    ## Return CRPS for all methods in this bootstrap
    crps_res
  }
  
  # Stop parallel backend
  stopCluster(cl)
  
  # Summarize results
  summary_results <- results %>%
    group_by(Method) %>%
    summarise(Mean_CRPS = mean(CRPS), Variance_CRPS = var(CRPS))
  
  # t-Test
  pairwise_t_test_results <- pairwise.t.test(
    x = results$CRPS, # CRPS values
    g = results$Method, # Grouping variable (method)
    pool.sd = FALSE,
    p.adjust.method = "holm" #
  )
    
  # Wilcoxon test
  pairwise_wilcox_results <- pairwise.wilcox.test(
    x = results$CRPS, 
    g = results$Method, 
    p.adjust.method = "holm"
  )
  
  return(list(detailed_crps_results = results, crps_results = summary_results, 
              pairwise_ttest = pairwise_t_test_results, pairwise_wilcoxtest = pairwise_wilcox_results))
}

#### Boston

Boston_new_results <- simulation_function(Boston_new, c(1:99/100), 30, seed = 123)
# print(Boston_new_results$detailed_crps_results)
print(Boston_new_results$crps_results)
print(Boston_new_results$pairwise_ttest)
print(Boston_new_results$pairwise_wilcoxtest)

#### Swiss

swiss_new_results <- simulation_function(swiss_new, c(1:99/100), 10, seed = 123)
print(swiss_new_results$crps_results)
print(swiss_new_results$pairwise_ttest)
print(swiss_new_results$pairwise_wilcoxtest)

#### Boston
imtCars_new_results <- simulation_function(mtcars_new, c(1:999/1000), 20, seed = 123)
print(mtCars_new_results$formatted_results)
