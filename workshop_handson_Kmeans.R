install.packages("tictoc", repos = "https://cloud.r-project.org")
# install.packages("fastDummies", repos = "https://cloud.r-project.org")

# Load required libraries
library(parallel)
library(tictoc)  # For timing
# library(fastDummies)  # For one-hot encoding categorical variables

patients_df <- read.csv("patients_100k.csv", stringsAsFactors = FALSE)

cols_for_clust <- c("RACE", "GENDER", "INCOME", "HEALTHCARE_EXPENSES", 
                    "HEALTHCARE_COVERAGE")

# Creating dummies
cluster_data <- patients_df[, cols_for_clust]
cluster_data$RACE <- as.factor(cluster_data$RACE)
cluster_data$GENDER <- as.factor(cluster_data$GENDER)
# cluster_data_dummy <- dummy_cols(cluster_data, 
#                                  select_columns = c("RACE", "GENDER"),
#                                  remove_first_dummy = TRUE,
#                                  remove_selected_columns = TRUE)
# build dummies (no intercept)
mm <- model.matrix(~ INCOME + HEALTHCARE_EXPENSES + HEALTHCARE_COVERAGE + RACE + GENDER,
                   data = cluster_data)
cluster_data_dummy <- as.data.frame(mm[, -1, drop = FALSE])

# scaling
cluster_data_scaled <- scale(cluster_data_dummy)

# Function to run K-means with specified number of cores
parallel_kmeans <- function(data, centers = 10, n_cores = 1, n_starts = 100) {
  
  # Ensure data is a matrix and check for finite values
  if (!is.matrix(data)) {
    data <- as.matrix(data)
  }
  
  # Double-check for non-finite values
  if (any(!is.finite(data))) {
    stop("Data contains non-finite values (NA/NaN/Inf)")
  }
  
  # Adjust centers if larger than number of unique rows
  unique_rows <- nrow(unique(data))
  if (centers > unique_rows) {
    cat("Warning: Reducing centers from", centers, "to", unique_rows, "due to data limitations\n")
    centers <- unique_rows
  }
  
  if (n_cores == 1) {
    # Sequential execution
    tryCatch({
      result <- kmeans(data, centers = centers, nstart = n_starts, iter.max = 100, algorithm = "Lloyd")
    }, error = function(e) {
      cat("Error in kmeans:", e$message, "\n")
      cat("Trying with fewer centers...\n")
      result <- kmeans(data, centers = min(5, centers), nstart = n_starts, iter.max = 100, algorithm = "Lloyd")
    })
  } else {
    # Parallel execution
    cl <- makeCluster(n_cores)
    
    # Export data to cluster
    clusterExport(cl, varlist = c("data", "centers"), envir = environment())
    
    # Run multiple K-means in parallel with error handling
    results <- parLapply(cl, 1:n_starts, function(i) {
      tryCatch({
        kmeans(data, centers = centers, nstart = 1, iter.max = 100, algorithm = "Lloyd")
      }, error = function(e) {
        # If error, try with fewer centers
        kmeans(data, centers = min(5, centers), nstart = 1, iter.max = 100, algorithm = "Lloyd")
      })
    })
    
    stopCluster(cl)
    
    # Select best result (lowest total within-cluster sum of squares)
    best_idx <- which.min(sapply(results, function(x) x$tot.withinss))
    result <- results[[best_idx]]
  }
  
  return(result)
}

tic("K-means with 1 core")
km_1core <- parallel_kmeans(cluster_data_scaled, centers = 5, n_cores = 1)
time_1core <- toc()
cat("Number of iterations:", km_1core$iter, "\n")
time_1core_val <- time_1core$toc - time_1core$tic


tic("K-means with 4 cores")
km_4cores <- parallel_kmeans(cluster_data_scaled, centers = 5, n_cores = 4)
time_4cores <- toc()
time_4cores_val <- time_4cores$toc - time_4cores$tic

speedup_4 <- time_1core_val / time_4cores_val
cat("\nSpeedup compared to 1 core:", round(speedup_4, 2), "x\n")
cat("Number of iterations:", km_4cores$iter, "\n")

### 5 vs 8 ###
# tic("K-means with 5 cores")
# km_5core <- parallel_kmeans(cluster_data_scaled, centers = 5, n_cores = 5)
# time_5core <- toc()
# cat("Number of iterations:", km_5core$iter, "\n")
# time_5core_val <- time_5core$toc - time_5core$tic
# 
# 
# tic("K-means with 8 cores")
# km_8cores <- parallel_kmeans(cluster_data_scaled, centers = 5, n_cores = 8)
# time_8cores <- toc()
# time_8cores_val <- time_8cores$toc - time_8cores$tic
# 
# speedup_8 <- time_5core_val / time_8cores_val
# cat("\nSpeedup compared to 5 cores:", round(speedup_8, 2), "x\n")
# cat("Number of iterations:", km_8cores$iter, "\n")