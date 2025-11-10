# install.packages("tictoc", repos = "https://cloud.r-project.org")
install.packages("randomForest", repos = "https://cloud.r-project.org")
install.packages("ranger", repos = "https://cloud.r-project.org")

# Load required libraries
library(parallel)
library(randomForest)
library(ranger)   # Fast implementation of Random Forest with built-in parallel support
library(tictoc)   # For timing

# Setting random seed for reproducibility
set.seed(13)

patients_df <- read.csv("patients_2020_100k.csv", stringsAsFactors = FALSE)
patients_df_clean <- na.omit(patients_df)

# Create encounters quartiles for classification
encounters_quartiles <- quantile(patients_df_clean$total_encounters, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
patients_df_clean$encounters_Class <- cut(patients_df_clean$total_encounters,
                                          breaks = c(-Inf, encounters_quartiles, Inf),
                                          labels = c("Low", "Medium_Low", "Medium_High", "High"))

# Select features for modeling
# We'll use both the original encounters for regression and encounters_Class for classification
feature_cols <- c("RACE", "GENDER", "STATE", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME", "total_conditions")

# Convert categorical variables to factors
patients_df_clean$RACE <- as.factor(patients_df_clean$RACE)
patients_df_clean$GENDER <- as.factor(patients_df_clean$GENDER)
patients_df_clean$STATE <- as.factor(patients_df_clean$STATE)

# Prepare features and targets
X <- patients_df_clean[, feature_cols]
y_regression <- patients_df_clean$total_encounters  # For regression (kept as in original)
y_classification <- patients_df_clean$encounters_Class  # For classification

# Split data into training and testing sets
train_indices <- sample(1:nrow(X), size = 0.8 * nrow(X))
test_indices <- setdiff(1:nrow(X), train_indices)

X_train <- X[train_indices, ]
X_test <- X[test_indices, ]
y_train_reg <- y_regression[train_indices]
y_test_reg <- y_regression[test_indices]
y_train_class <- y_classification[train_indices]
y_test_class <- y_classification[test_indices]

# For larger datasets, we can duplicate the data to show performance differences
# n_duplicates <- 10  # Adjust based on your system
# X_train_large <- X_train[rep(seq_len(nrow(X_train)), n_duplicates), ]
# y_train_reg_large <- y_train_reg[rep(seq_len(length(y_train_reg)), n_duplicates)]
# y_train_class_large <- y_train_class[rep(seq_len(length(y_train_class)), n_duplicates)]

# For now, use original size
X_train_large <- X_train
y_train_reg_large <- y_train_reg
y_train_class_large <- y_train_class

# Function to run Random Forest with specified number of cores using randomForest package
parallel_rf_manual <- function(X, y, n_cores = 1, ntree = 500, classification = TRUE) {
  if (n_cores == 1) {
    # Sequential execution
    if (classification) {
      model <- randomForest(x = X, y = y, ntree = ntree, importance = TRUE)
    } else {
      model <- randomForest(x = X, y = y, ntree = ntree, importance = TRUE)
    }
  } else {
    # Parallel execution - split trees across cores
    trees_per_core <- ceiling(ntree / n_cores)
    
    # Create cluster
    cl <- makeCluster(n_cores)
    
    # Export necessary objects and load library on each worker
    clusterEvalQ(cl, library(randomForest))
    clusterExport(cl, varlist = c("X", "y", "trees_per_core", "classification"), 
                  envir = environment())
    
    # Train Random Forest models in parallel
    rf_list <- parLapply(cl, 1:n_cores, function(i) {
      set.seed(i * 1000)  # Different seed for each worker
      if (classification) {
        randomForest(x = X, y = y, ntree = trees_per_core, importance = FALSE)
      } else {
        randomForest(x = X, y = y, ntree = trees_per_core, importance = FALSE)
      }
    })
    
    stopCluster(cl)
    
    # Combine forests using randomForest's combine function
    model <- do.call(randomForest::combine, rf_list)
  }
  
  return(model)
}

# Function to run Random Forest using ranger (has built-in parallel support)
parallel_rf_ranger <- function(X, y, n_cores = 1, num_trees = 500, classification = TRUE) {
  # Combine X and y for ranger
  data_combined <- cbind(X, target = y)
  
  if (classification) {
    model <- ranger(target ~ ., 
                    data = data_combined, 
                    num.trees = num_trees,
                    importance = "impurity",
                    num.threads = n_cores,
                    classification = TRUE)
  } else {
    model <- ranger(target ~ ., 
                    data = data_combined, 
                    num.trees = num_trees,
                    importance = "impurity",
                    num.threads = n_cores)
  }
  
  return(model)
}

# ----- Experiments (classification) -----

tic("Random Forest Classification with 1 core (randomForest)")
rf_1core <- parallel_rf_manual(X_train_large, y_train_class_large, 
                               n_cores = 1, ntree = 200, classification = TRUE)
time_1core <- toc()
time_1core_val <- time_1core$toc - time_1core$tic
# Evaluate model
pred_1core <- predict(rf_1core, X_test)
accuracy_1core <- mean(pred_1core == y_test_class)
cat("\nAccuracy:", round(accuracy_1core * 100, 2), "%\n")

tic("Random Forest Classification with 4 cores (randomForest)")
rf_4cores <- parallel_rf_manual(X_train_large, y_train_class_large, 
                                n_cores = 4, ntree = 200, classification = TRUE)
time_4cores <- toc()
time_4cores_val <- time_4cores$toc - time_4cores$tic

speedup_4 <- time_1core_val / time_4cores_val
cat("\nSpeedup compared to 1 core:", round(speedup_4, 2), "x\n")
pred_4cores <- predict(rf_4cores, X_test)
accuracy_4cores <- mean(pred_4cores == y_test_class)
cat("Accuracy:", round(accuracy_4cores * 100, 2), "%\n")

cat("\n=== RANGER PACKAGE COMPARISON ===\n")
# Test with different core counts using ranger
ranger_times <- c()
ranger_cores <- c()

for (n_cores in c(1, 4)) {
  tic(paste("Ranger with", n_cores, "cores"))
  rf_ranger <- parallel_rf_ranger(X_train_large, y_train_class_large, 
                                  n_cores = n_cores, num_trees = 200, 
                                  classification = TRUE)
  time_ranger <- toc()
  ranger_times <- c(ranger_times, time_ranger$toc - time_ranger$tic)
  ranger_cores <- c(ranger_cores, n_cores)
  
  # Calculate accuracy
  pred_ranger <- predict(rf_ranger, data = cbind(X_test, target = y_test_class))$predictions
  accuracy_ranger <- mean(pred_ranger == y_test_class)
  cat("Accuracy:", round(accuracy_ranger * 100, 2), "%\n\n")
}