# # F1 Race Result Classification Model
# # This script builds a simple classification model to predict F1 race outcomes

# # Install and load required packages (Fixed version)
# required_packages <- c("randomForest", "caret", "e1071", "rpart")

# # Install missing packages
# for(package in required_packages) {
#   if(!require(package, character.only = TRUE)) {
#     install.packages(package, dependencies = TRUE)
#     library(package, character.only = TRUE)
#   }
# }

# cat("All required packages loaded successfully!\n")

# # ==============================================================================
# # 1. DATA LOADING AND EXPLORATION
# # ==============================================================================

# # Download F1 dataset from Kaggle
# # You can download from: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
# # For this example, we'll simulate loading the data
# # Replace 'your_path' with actual path to downloaded CSV files

# # Load the main datasets
# cat("Loading F1 datasets...\n")

# # Example data loading (replace with your actual file paths)
# # races <- read_csv("races.csv")
# # results <- read_csv("results.csv")
# # drivers <- read_csv("drivers.csv")
# # constructors <- read_csv("constructors.csv")

# # For demonstration, let's create a sample dataset
# # In practice, you would merge the actual F1 datasets

# set.seed(123)
# n_races <- 1000

# # Create sample F1 data
# f1_data <- data.frame(
#   race_id = 1:n_races,
#   driver_age = sample(18:45, n_races, replace = TRUE),
#   grid_position = sample(1:20, n_races, replace = TRUE),
#   constructor_points = sample(0:500, n_races, replace = TRUE),
#   driver_experience = sample(0:20, n_races, replace = TRUE),
#   weather_condition = sample(c("Sunny", "Rainy", "Cloudy"), n_races, replace = TRUE),
#   track_difficulty = sample(1:10, n_races, replace = TRUE),
#   qualifying_time = runif(n_races, 60, 90),
#   previous_race_points = sample(0:25, n_races, replace = TRUE),
#   # Target variable: Race finish position category
#   finish_category = sample(c("Podium", "Points", "No_Points"), n_races, 
#                           replace = TRUE, prob = c(0.15, 0.35, 0.5))
# )

# # Convert categorical variables to factors
# f1_data$weather_condition <- as.factor(f1_data$weather_condition)
# f1_data$finish_category <- as.factor(f1_data$finish_category)

# cat("Data loaded successfully!\n")
# cat("Dataset dimensions:", dim(f1_data), "\n")

# # ==============================================================================
# # 2. EXPLORATORY DATA ANALYSIS
# # ==============================================================================

# cat("\n=== EXPLORATORY DATA ANALYSIS ===\n")

# # Basic data summary
# print(summary(f1_data))

# # Check for missing values
# cat("\nMissing values:\n")
# missing_counts <- sapply(f1_data, function(x) sum(is.na(x)))
# print(missing_counts)

# # Target variable distribution
# cat("\nTarget variable distribution:\n")
# print(table(f1_data$finish_category))

# # Correlation matrix for numeric variables
# numeric_cols <- sapply(f1_data, is.numeric)
# numeric_data <- f1_data[, numeric_cols]
# numeric_data <- numeric_data[, !names(numeric_data) %in% "race_id"]
# correlation_matrix <- cor(numeric_data)

# cat("\nCorrelation matrix created\n")

# # ==============================================================================
# # 3. DATA PREPROCESSING
# # ==============================================================================

# cat("\n=== DATA PREPROCESSING ===\n")

# # Remove ID column for modeling
# modeling_data <- f1_data[, !names(f1_data) %in% "race_id"]

# # Handle any missing values (if they exist)
# if(sum(is.na(modeling_data)) > 0) {
#   modeling_data <- na.omit(modeling_data)
#   cat("Missing values removed\n")
# }

# # Create dummy variables for categorical predictors
# modeling_data_encoded <- model.matrix(finish_category ~ ., data = modeling_data)[,-1]
# modeling_data_encoded <- data.frame(modeling_data_encoded)
# modeling_data_encoded$finish_category <- modeling_data$finish_category

# cat("Data preprocessing completed\n")
# cat("Final dataset dimensions:", dim(modeling_data_encoded), "\n")

# # ==============================================================================
# # 4. TRAIN-TEST SPLIT
# # ==============================================================================

# cat("\n=== SPLITTING DATA ===\n")

# set.seed(123)
# train_index <- createDataPartition(modeling_data_encoded$finish_category, 
#                                   p = 0.8, list = FALSE)
# train_data <- modeling_data_encoded[train_index, ]
# test_data <- modeling_data_encoded[-train_index, ]

# cat("Training set size:", nrow(train_data), "\n")
# cat("Test set size:", nrow(test_data), "\n")

# # ==============================================================================
# # 5. MODEL BUILDING
# # ==============================================================================

# cat("\n=== BUILDING CLASSIFICATION MODELS ===\n")

# # Set up cross-validation
# ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# # Model 1: Random Forest
# cat("Training Random Forest model...\n")
# rf_model <- train(finish_category ~ ., 
#                   data = train_data,
#                   method = "rf",
#                   trControl = ctrl,
#                   tuneLength = 3)

# # Model 2: Decision Tree
# cat("Training Decision Tree model...\n")
# dt_model <- train(finish_category ~ ., 
#                   data = train_data,
#                   method = "rpart",
#                   trControl = ctrl,
#                   tuneLength = 3)

# # Model 3: Support Vector Machine
# cat("Training SVM model...\n")
# svm_model <- train(finish_category ~ ., 
#                    data = train_data,
#                    method = "svmRadial",
#                    trControl = ctrl,
#                    tuneLength = 3)

# cat("All models trained successfully!\n")

# # ==============================================================================
# # 6. MODEL EVALUATION
# # ==============================================================================

# cat("\n=== MODEL EVALUATION ===\n")

# # Make predictions
# rf_pred <- predict(rf_model, test_data)
# dt_pred <- predict(dt_model, test_data)
# svm_pred <- predict(svm_model, test_data)

# # Calculate accuracy
# rf_accuracy <- confusionMatrix(rf_pred, test_data$finish_category)$overall['Accuracy']
# dt_accuracy <- confusionMatrix(dt_pred, test_data$finish_category)$overall['Accuracy']
# svm_accuracy <- confusionMatrix(svm_pred, test_data$finish_category)$overall['Accuracy']

# # Create results summary
# results_summary <- data.frame(
#   Model = c("Random Forest", "Decision Tree", "Support Vector Machine"),
#   Accuracy = c(rf_accuracy, dt_accuracy, svm_accuracy)
# )

# cat("Model Performance Summary:\n")
# print(results_summary)

# # Detailed confusion matrix for best model
# best_model_name <- results_summary$Model[which.max(results_summary$Accuracy)]
# cat(paste("\nDetailed results for best model:", best_model_name, "\n"))

# if(best_model_name == "Random Forest") {
#   best_predictions <- rf_pred
#   best_model <- rf_model
# } else if(best_model_name == "Decision Tree") {
#   best_predictions <- dt_pred
#   best_model <- dt_model
# } else {
#   best_predictions <- svm_pred
#   best_model <- svm_model
# }

# print(confusionMatrix(best_predictions, test_data$finish_category))

# # ==============================================================================
# # 7. FEATURE IMPORTANCE (for Random Forest)
# # ==============================================================================

# if(exists("rf_model")) {
#   cat("\n=== FEATURE IMPORTANCE ===\n")
#   importance_scores <- varImp(rf_model)
#   cat("Top features for F1 race outcome prediction:\n")
#   print(importance_scores)
# }

# # ==============================================================================
# # 8. MAKING NEW PREDICTIONS
# # ==============================================================================

# cat("\n=== EXAMPLE PREDICTIONS ===\n")

# # Create sample new data for prediction
# new_race_data <- data.frame(
#   driver_age = c(25, 35),
#   grid_position = c(3, 15),
#   constructor_points = c(400, 100),
#   driver_experience = c(5, 12),
#   weather_conditionRainy = c(0, 1),
#   weather_conditionSunny = c(1, 0),
#   track_difficulty = c(7, 4),
#   qualifying_time = c(65.5, 78.2),
#   previous_race_points = c(18, 2)
# )

# # Make predictions with the best model
# sample_predictions <- predict(best_model, new_race_data)
# sample_probabilities <- predict(best_model, new_race_data, type = "prob")

# cat("Sample predictions:\n")
# for(i in 1:nrow(new_race_data)) {
#   cat(paste("Race", i, "- Predicted outcome:", sample_predictions[i], "\n"))
# }

# # ==============================================================================
# # 9. MODEL SAVING
# # ==============================================================================

# cat("\n=== SAVING MODEL ===\n")

# # Save the best model
# saveRDS(best_model, "best_f1_model.rds")
# cat("Best model saved as 'best_f1_model.rds'\n")

# # Save the results summary
# write.csv(results_summary, "model_performance.csv", row.names = FALSE)
# cat("Results saved as 'model_performance.csv'\n")

# cat("\n=== F1 CLASSIFICATION MODEL COMPLETE ===\n")
# cat("Summary:\n")
# cat("- Dataset size:", nrow(f1_data), "races\n")
# cat("- Best model:", best_model_name, "\n")
# cat("- Best accuracy:", round(max(results_summary$Accuracy), 4), "\n")
# cat("- Model saved for future use\n")

# # ==============================================================================
# # 10. ADDITIONAL ANALYSIS (OPTIONAL)
# # ==============================================================================
# # Fixed Visualization Code for F1 Classification Results
# # Run this after your main model has completed

# # Create model comparison data frame
# model_comparison <- data.frame(
#   Model = c("Random Forest", "Decision Tree", "Support Vector Machine"),
#   Accuracy = c(rf_accuracy, dt_accuracy, svm_accuracy)
# )

# # Display the results
# print("=== MODEL COMPARISON RESULTS ===")
# print(model_comparison)

# # Create a simple barplot
# barplot(model_comparison$Accuracy, 
#         names.arg = model_comparison$Model,
#         main = "F1 Classification Model Comparison",
#         ylab = "Accuracy", 
#         col = c("blue", "green", "red"),
#         ylim = c(0, 1),
#         las = 2)  # Rotate x-axis labels

# # Add accuracy values on top of bars
# text(x = 1:3, 
#      y = model_comparison$Accuracy + 0.02, 
#      labels = round(model_comparison$Accuracy, 3),
#      pos = 3)

# # Alternative: Horizontal barplot (better for long model names)
# cat("\n=== Alternative Horizontal Plot ===\n")
# par(mar = c(5, 8, 4, 2))  # Increase left margin
# barplot(model_comparison$Accuracy, 
#         names.arg = model_comparison$Model,
#         main = "F1 Classification Model Comparison",
#         xlab = "Accuracy", 
#         col = c("lightblue", "lightgreen", "lightcoral"),
#         xlim = c(0, 1),
#         horiz = TRUE)

# # Add grid for better readability
# grid(nx = NULL, ny = 0, col = "gray", lty = "dotted")

# # Reset margins
# par(mar = c(5, 4, 4, 2))

# # Print best model information
# best_model_index <- which.max(model_comparison$Accuracy)
# best_model_name <- model_comparison$Model[best_model_index]
# best_accuracy <- model_comparison$Accuracy[best_model_index]

# cat("\n=== FINAL RESULTS ===\n")
# cat("Best performing model:", best_model_name, "\n")
# cat("Best accuracy:", round(best_accuracy, 4), "\n")
# cat("Model improvement over random guessing:", round((best_accuracy - 0.33) * 100, 2), "percentage points\n")

# # Create a summary table
# cat("\n=== DETAILED PERFORMANCE ===\n")
# performance_table <- data.frame(
#   Model = model_comparison$Model,
#   Accuracy = round(model_comparison$Accuracy, 4),
#   Rank = rank(-model_comparison$Accuracy)
# )
# print(performance_table)

# # Save the visualization results
# write.csv(performance_table, "f1_model_detailed_results.csv", row.names = FALSE)
# cat("\nDetailed results saved to 'f1_model_detailed_results.csv'\n")


# F1 Race Result Classification Model
# This script builds a simple classification model to predict F1 race outcomes

# Install and load required packages (Fixed version)
required_packages <- c("randomForest", "caret", "e1071", "rpart")

# Install missing packages
for(package in required_packages) {
  if(!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

cat("All required packages loaded successfully!\n")

# ==============================================================================
# 1. DATA LOADING FROM KAGGLE F1 DATASET
# ==============================================================================

# INSTRUCTIONS TO DOWNLOAD KAGGLE F1 DATASET:
# 1. Go to: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
# 2. Click "Download" (you may need to create a free Kaggle account)
# 3. Extract the ZIP file to your R working directory
# 4. You should have these CSV files: races.csv, results.csv, drivers.csv, constructors.csv, etc.

cat("Loading Real F1 Dataset from Kaggle...\n")

# Check if files exist
required_files <- c("races.csv", "results.csv", "drivers.csv", "constructors.csv", "qualifying.csv")
missing_files <- required_files[!file.exists(required_files)]

if(length(missing_files) > 0) {
  cat("❌ Missing files:", paste(missing_files, collapse = ", "), "\n")
  cat("📥 Please download the F1 dataset from Kaggle:\n")
  cat("   https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020\n")
  cat("📁 Extract all CSV files to your current working directory:\n")
  cat("   Current directory:", getwd(), "\n")
  cat("\n🔄 For now, using simulated data for demonstration...\n\n")
  
  # Use simulated data as fallback
  set.seed(123)
  n_races <- 1000
  
  # Create realistic sample data
  f1_data <- data.frame(
    raceId = 1:n_races,
    year = sample(2020:2023, n_races, replace = TRUE),
    driverId = sample(1:50, n_races, replace = TRUE),
    constructorId = sample(1:10, n_races, replace = TRUE),
    grid = sample(1:20, n_races, replace = TRUE),
    positionOrder = sample(1:20, n_races, replace = TRUE),
    points = c(rep(25, n_races*0.05), rep(18, n_races*0.05), rep(15, n_races*0.05), 
               rep(12, n_races*0.05), rep(10, n_races*0.05), rep(8, n_races*0.05),
               rep(6, n_races*0.05), rep(4, n_races*0.05), rep(2, n_races*0.05), 
               rep(1, n_races*0.05), rep(0, n_races*0.5)),
    fastestLapTime_ms = sample(80000:95000, n_races, replace = TRUE),
    driver_name = sample(c("Hamilton", "Verstappen", "Leclerc", "Russell", "Sainz"), n_races, replace = TRUE),
    constructor_name = sample(c("Mercedes", "Red Bull", "Ferrari", "McLaren", "Alpine"), n_races, replace = TRUE),
    race_winner = ifelse(sample(1:20, n_races, replace = TRUE) == 1, 1, 0)
  )
  
} else {
  # Load actual Kaggle F1 data
  cat("✅ Loading real F1 data files...\n")
  
  races <- read.csv("races.csv", stringsAsFactors = FALSE)
  results <- read.csv("results.csv", stringsAsFactors = FALSE)
  drivers <- read.csv("drivers.csv", stringsAsFactors = FALSE)
  constructors <- read.csv("constructors.csv", stringsAsFactors = FALSE)
  
  # Load qualifying data if available
  if(file.exists("qualifying.csv")) {
    qualifying <- read.csv("qualifying.csv", stringsAsFactors = FALSE)
  }
  
  cat("📊 Dataset sizes:\n")
  cat("   Races:", nrow(races), "\n")
  cat("   Results:", nrow(results), "\n")
  cat("   Drivers:", nrow(drivers), "\n")
  cat("   Constructors:", nrow(constructors), "\n")
  
  # ==============================================================================
  # DATA PREPROCESSING AND FEATURE ENGINEERING
  # ==============================================================================
  
  cat("\n🔧 Creating features from real F1 data...\n")
  
  # Merge datasets to create comprehensive features
  f1_data <- results %>%
    merge(races[, c("raceId", "year", "circuitId", "name")], by = "raceId") %>%
    merge(drivers[, c("driverId", "driverRef", "forename", "surname", "dob", "nationality")], by = "driverId") %>%
    merge(constructors[, c("constructorId", "constructorRef", "name")], by = "constructorId") %>%
    # Filter for modern era (2010 onwards) for better relevance
    filter(year >= 2010) %>%
    # Create driver full name
    mutate(
      driver_name = paste(forename, surname),
      constructor_name = name.y,
      race_name = name.x,
      # Calculate driver age at time of race
      driver_age = as.numeric(as.Date(paste(year, "06", "01", sep="-")) - as.Date(dob)) / 365.25,
      # Convert position to numeric (handle "\\N" values)
      positionOrder = ifelse(positionOrder == "\\N", 21, as.numeric(positionOrder)),
      # Create winner indicator (1st position = winner)
      race_winner = ifelse(positionOrder == 1, 1, 0),
      # Handle grid position
      grid = ifelse(grid == 0, 21, grid),  # 0 means started from pit lane
      # Convert points to numeric
      points = ifelse(points == "\\N", 0, as.numeric(points)),
      # Handle fastest lap time
      fastestLapTime_ms = ifelse(fastestLapTime == "\\N", NA, 
                                as.numeric(substr(gsub(":", "", fastestLapTime), 1, 6))),
      # Create experience feature (races completed before this race)
      driver_experience = ave(raceId, driverId, FUN = function(x) seq_along(x) - 1)
    ) %>%
    # Select relevant columns
    select(raceId, year, circuitId, driverId, constructorId, driver_name, constructor_name,
           driver_age, grid, positionOrder, points, fastestLapTime_ms, driver_experience, 
           race_winner, driverRef, constructorRef) %>%
    # Remove rows with missing critical data
    filter(!is.na(driver_age), !is.na(grid), !is.na(positionOrder)) %>%
    # Create additional features
    mutate(
      # Constructor performance (avg points in last 5 races)
      constructor_recent_performance = ave(points, constructorId, 
                                          FUN = function(x) {
                                            n <- length(x)
                                            sapply(1:n, function(i) {
                                              start <- max(1, i-4)
                                              mean(x[start:(i-1)], na.rm = TRUE)
                                            })
                                          }),
      # Driver recent form
      driver_recent_performance = ave(points, driverId,
                                     FUN = function(x) {
                                       n <- length(x)
                                       sapply(1:n, function(i) {
                                         start <- max(1, i-4)
                                         mean(x[start:(i-1)], na.rm = TRUE)
                                       })
                                     }),
      # Track-specific features
      circuit_difficulty = ave(positionOrder, circuitId, FUN = function(x) sd(x, na.rm = TRUE))
    ) %>%
    # Handle NaN values from initial races
    mutate(
      constructor_recent_performance = ifelse(is.nan(constructor_recent_performance), 0, constructor_recent_performance),
      driver_recent_performance = ifelse(is.nan(driver_recent_performance), 0, driver_recent_performance),
      circuit_difficulty = ifelse(is.nan(circuit_difficulty), 5, circuit_difficulty)
    )
  
  cat("✅ Feature engineering completed!\n")
  cat("📈 Final dataset size:", nrow(f1_data), "race results\n")
  cat("🏆 Winners in dataset:", sum(f1_data$race_winner), "\n")
  cat("📊 Win rate:", round(mean(f1_data$race_winner) * 100, 2), "%\n")
}

# Convert categorical variables to factors
f1_data$driver_name <- as.factor(f1_data$driver_name)
f1_data$constructor_name <- as.factor(f1_data$constructor_name)
f1_data$race_winner <- as.factor(f1_data$race_winner)
levels(f1_data$race_winner) <- c("No_Win", "Winner")

cat("Data loaded successfully!\n")
cat("Dataset dimensions:", dim(f1_data), "\n")

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ==============================================================================

cat("\n=== EXPLORATORY DATA ANALYSIS ===\n")

# Basic data summary
print(summary(f1_data))

# Check for missing values
cat("\nMissing values:\n")
missing_counts <- sapply(f1_data, function(x) sum(is.na(x)))
print(missing_counts)

# Target variable distribution
cat("\nRace Winner Distribution:\n")
print(table(f1_data$race_winner))
cat("\nWinner percentage:", round(prop.table(table(f1_data$race_winner))[2] * 100, 1), "%\n")

# Correlation matrix for numeric variables
numeric_cols <- sapply(f1_data, is.numeric)
numeric_data <- f1_data[, numeric_cols]
# Remove ID columns from correlation
id_columns <- c("raceId", "driverId", "constructorId", "circuitId", "year")
numeric_data <- numeric_data[, !names(numeric_data) %in% id_columns]
correlation_matrix <- cor(numeric_data, use = "complete.obs")

cat("\nCorrelation matrix created for", ncol(numeric_data), "numeric features\n")

# ==============================================================================
# 3. DATA PREPROCESSING
# ==============================================================================

cat("\n=== DATA PREPROCESSING ===\n")

# Remove ID columns and select features for modeling
modeling_data <- f1_data[, !names(f1_data) %in% c("raceId", "driverId", "constructorId", 
                                                   "circuitId", "year", "driverRef", 
                                                   "constructorRef", "positionOrder")]

# Handle any remaining missing values
if(sum(is.na(modeling_data)) > 0) {
  cat("Handling missing values...\n")
  # For numeric variables, use median imputation
  numeric_columns <- sapply(modeling_data, is.numeric)
  for(col in names(modeling_data)[numeric_columns]) {
    if(sum(is.na(modeling_data[, col])) > 0) {
      median_val <- median(modeling_data[, col], na.rm = TRUE)
      modeling_data[is.na(modeling_data[, col]), col] <- median_val
    }
  }
  cat("Missing values handled\n")
}

# Limit factor levels to prevent too many dummy variables
if("driver_name" %in% names(modeling_data)) {
  # Keep only top 15 drivers by race count
  top_drivers <- names(sort(table(modeling_data$driver_name), decreasing = TRUE))[1:15]
  modeling_data$driver_name <- ifelse(modeling_data$driver_name %in% top_drivers, 
                                     as.character(modeling_data$driver_name), "Other")
  modeling_data$driver_name <- as.factor(modeling_data$driver_name)
}

if("constructor_name" %in% names(modeling_data)) {
  # Keep only top 10 constructors
  top_constructors <- names(sort(table(modeling_data$constructor_name), decreasing = TRUE))[1:10]
  modeling_data$constructor_name <- ifelse(modeling_data$constructor_name %in% top_constructors,
                                          as.character(modeling_data$constructor_name), "Other")
  modeling_data$constructor_name <- as.factor(modeling_data$constructor_name)
}

# Create dummy variables for categorical predictors
modeling_data_encoded <- model.matrix(race_winner ~ ., data = modeling_data)[,-1]
modeling_data_encoded <- data.frame(modeling_data_encoded)
modeling_data_encoded$race_winner <- modeling_data$race_winner

cat("Data preprocessing completed\n")
cat("Final dataset dimensions:", dim(modeling_data_encoded), "\n")

# ==============================================================================
# 4. TRAIN-TEST SPLIT
# ==============================================================================

cat("\n=== SPLITTING DATA ===\n")

set.seed(123)
train_index <- createDataPartition(modeling_data_encoded$race_winner, 
                                  p = 0.8, list = FALSE)
train_data <- modeling_data_encoded[train_index, ]
test_data <- modeling_data_encoded[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# ==============================================================================
# 5. MODEL BUILDING
# ==============================================================================

cat("\n=== BUILDING CLASSIFICATION MODELS ===\n")

# Set up cross-validation
ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# Model 1: Random Forest for Winner Prediction
cat("Training Random Forest model for winner prediction...\n")
rf_model <- train(race_winner ~ ., 
                  data = train_data,
                  method = "rf",
                  trControl = ctrl,
                  tuneLength = 3)

# Model 2: Decision Tree for Winner Prediction
cat("Training Decision Tree model for winner prediction...\n")
dt_model <- train(race_winner ~ ., 
                  data = train_data,
                  method = "rpart",
                  trControl = ctrl,
                  tuneLength = 3)

# Model 3: Support Vector Machine for Winner Prediction
cat("Training SVM model for winner prediction...\n")
svm_model <- train(race_winner ~ ., 
                   data = train_data,
                   method = "svmRadial",
                   trControl = ctrl,
                   tuneLength = 3)

cat("All models trained successfully!\n")

# ==============================================================================
# 6. MODEL EVALUATION
# ==============================================================================

cat("\n=== MODEL EVALUATION ===\n")

# Make predictions
rf_pred <- predict(rf_model, test_data)
dt_pred <- predict(dt_model, test_data)
svm_pred <- predict(svm_model, test_data)

# Calculate accuracy
rf_accuracy <- confusionMatrix(rf_pred, test_data$race_winner)$overall['Accuracy']
dt_accuracy <- confusionMatrix(dt_pred, test_data$race_winner)$overall['Accuracy']
svm_accuracy <- confusionMatrix(svm_pred, test_data$race_winner)$overall['Accuracy']

# Additional metrics for winner prediction
rf_conf <- confusionMatrix(rf_pred, test_data$race_winner)
dt_conf <- confusionMatrix(dt_pred, test_data$race_winner)
svm_conf <- confusionMatrix(svm_pred, test_data$race_winner)

# Create results summary
results_summary <- data.frame(
  Model = c("Random Forest", "Decision Tree", "Support Vector Machine"),
  Accuracy = c(rf_accuracy, dt_accuracy, svm_accuracy),
  Winner_Precision = c(
    rf_conf$byClass['Pos Pred Value'], 
    dt_conf$byClass['Pos Pred Value'], 
    svm_conf$byClass['Pos Pred Value']
  ),
  Winner_Recall = c(
    rf_conf$byClass['Sensitivity'], 
    dt_conf$byClass['Sensitivity'], 
    svm_conf$byClass['Sensitivity']
  )
)

cat("F1 Race Winner Prediction Performance:\n")
print(results_summary)

# Detailed confusion matrix for best model
best_model_name <- results_summary$Model[which.max(results_summary$Accuracy)]
cat(paste("\nDetailed results for best model:", best_model_name, "\n"))

if(best_model_name == "Random Forest") {
  best_predictions <- rf_pred
  best_model <- rf_model
} else if(best_model_name == "Decision Tree") {
  best_predictions <- dt_pred
  best_model <- dt_model
} else {
  best_predictions <- svm_pred
  best_model <- svm_model
}

print(confusionMatrix(best_predictions, test_data$race_winner))

# ==============================================================================
# 7. FEATURE IMPORTANCE (for Random Forest)
# ==============================================================================

if(exists("rf_model")) {
  cat("\n=== FEATURE IMPORTANCE ===\n")
  importance_scores <- varImp(rf_model)
  cat("Top features for F1 race outcome prediction:\n")
  print(importance_scores)
}

# ==============================================================================
# 8. WINNER PREDICTION FOR UPCOMING RACES
# ==============================================================================

cat("\n=== PREDICTING RACE WINNERS ===\n")

# Get the exact column names from the training data (excluding target variable)
model_features <- names(train_data)[names(train_data) != "race_winner"]
cat("Model expects these features:", length(model_features), "\n")
cat("Features:", paste(head(model_features, 10), collapse = ", "), "...\n")

# Create prediction data that matches exactly what the model expects
create_prediction_row <- function(driver_name, constructor_name, grid_pos, driver_age, 
                                 driver_exp, recent_perf = 10, constructor_perf = 12) {
  
  # Initialize with all zeros/defaults
  pred_row <- data.frame(matrix(0, nrow = 1, ncol = length(model_features)))
  names(pred_row) <- model_features
  
  # Set numeric features
  if("driver_age" %in% model_features) pred_row$driver_age <- driver_age
  if("grid" %in% model_features) pred_row$grid <- grid_pos
  if("driver_experience" %in% model_features) pred_row$driver_experience <- driver_exp
  if("points" %in% model_features) pred_row$points <- 0  # Current race points (unknown)
  if("driver_recent_performance" %in% model_features) pred_row$driver_recent_performance <- recent_perf
  if("constructor_recent_performance" %in% model_features) pred_row$constructor_recent_performance <- constructor_perf
  if("circuit_difficulty" %in% model_features) pred_row$circuit_difficulty <- 5
  if("fastestLapTime_ms" %in% model_features) pred_row$fastestLapTime_ms <- 85000
  
  # Set driver dummy variables
  driver_col <- paste0("driver_name", driver_name)
  if(driver_col %in% model_features) pred_row[, driver_col] <- 1
  
  # Set constructor dummy variables  
  constructor_col <- paste0("constructor_name", constructor_name)
  if(constructor_col %in% model_features) pred_row[, constructor_col] <- 1
  
  return(pred_row)
}

# Create realistic race scenarios
scenarios <- list(
  list(name = "Hamilton - Monaco GP", driver = "Lewis Hamilton", constructor = "Mercedes", 
       grid = 3, age = 39, exp = 310, recent = 15, const_perf = 20),
  list(name = "Verstappen - Silverstone GP", driver = "Max Verstappen", constructor = "Red Bull", 
       grid = 1, age = 26, exp = 150, recent = 22, const_perf = 25),
  list(name = "Leclerc - Monza GP", driver = "Charles Leclerc", constructor = "Ferrari", 
       grid = 2, age = 26, exp = 120, recent = 12, const_perf = 18),
  list(name = "Russell - Spa GP", driver = "George Russell", constructor = "Mercedes", 
       grid = 4, age = 26, exp = 80, recent = 8, const_perf = 20)
)

cat("\n🏎️ F1 RACE WINNER PREDICTIONS 🏎️\n")
cat("=====================================\n")

prediction_results <- data.frame(
  Scenario = character(),
  Driver = character(),
  Prediction = character(),
  Win_Probability = numeric(),
  stringsAsFactors = FALSE
)

for(i in 1:length(scenarios)) {
  scenario <- scenarios[[i]]
  
  # Create prediction data
  tryCatch({
    pred_data <- create_prediction_row(
      driver_name = scenario$driver,
      constructor_name = scenario$constructor,
      grid_pos = scenario$grid,
      driver_age = scenario$age,
      driver_exp = scenario$exp,
      recent_perf = scenario$recent,
      constructor_perf = scenario$const_perf
    )
    
    # Make prediction
    prediction <- predict(best_model, pred_data)
    probabilities <- predict(best_model, pred_data, type = "prob")
    
    win_prob <- if("Winner" %in% colnames(probabilities)) probabilities[1, "Winner"] else 0.5
    
    # Store results
    prediction_results <- rbind(prediction_results, data.frame(
      Scenario = scenario$name,
      Driver = scenario$driver,
      Prediction = as.character(prediction[1]),
      Win_Probability = round(win_prob * 100, 1),
      stringsAsFactors = FALSE
    ))
    
    # Print results
    cat(paste("🏁", scenario$name, "\n"))
    cat(paste("   Driver:", scenario$driver, "\n"))
    cat(paste("   Prediction:", ifelse(prediction[1] == "Winner", "WILL WIN! 🏆", "Won't win"), "\n"))
    cat(paste("   Win Probability:", round(win_prob * 100, 1), "%\n"))
    cat("   ---------------------\n")
    
  }, error = function(e) {
    cat("❌ Error predicting for", scenario$name, ":", e$message, "\n")
    cat("   ---------------------\n")
  })
}

# Find most likely winner
if(nrow(prediction_results) > 0) {
  best_prediction <- prediction_results[which.max(prediction_results$Win_Probability), ]
  cat(paste("\n🎯 MOST LIKELY WINNER:", best_prediction$Scenario, "\n"))
  cat(paste("   Driver:", best_prediction$Driver, "\n"))
  cat(paste("   Win Probability:", best_prediction$Win_Probability, "%\n"))
  
  # Save prediction results
  write.csv(prediction_results, "f1_winner_predictions.csv", row.names = FALSE)
  cat("\n💾 Predictions saved to 'f1_winner_predictions.csv'\n")
} else {
  cat("\n❌ No successful predictions made. Check model features.\n")
}

# ==============================================================================
# 9. MODEL SAVING
# ==============================================================================

cat("\n=== SAVING MODEL ===\n")

# Save the best model
saveRDS(best_model, "best_f1_model.rds")
cat("Best model saved as 'best_f1_model.rds'\n")

# Save the results summary
write.csv(results_summary, "model_performance.csv", row.names = FALSE)
cat("Results saved as 'model_performance.csv'\n")

cat("\n=== F1 CLASSIFICATION MODEL COMPLETE ===\n")
cat("Summary:\n")
cat("- Dataset size:", nrow(f1_data), "races\n")
cat("- Best model:", best_model_name, "\n")
cat("- Best accuracy:", round(max(results_summary$Accuracy), 4), "\n")
cat("- Model saved for future use\n")

# ==============================================================================
# 10. ADDITIONAL ANALYSIS (OPTIONAL)
# ==============================================================================

# ==============================================================================
# 10. VISUALIZATION OF RESULTS
# ==============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n")

# Create model comparison data frame
model_comparison <- data.frame(
  Model = c("Random Forest", "Decision Tree", "Support Vector Machine"),
  Accuracy = c(rf_accuracy, dt_accuracy, svm_accuracy)
)

# Display the results
cat("=== MODEL COMPARISON RESULTS ===\n")
print(model_comparison)

# Create a simple barplot
barplot(model_comparison$Accuracy, 
        names.arg = model_comparison$Model,
        main = "F1 Classification Model Comparison",
        ylab = "Accuracy", 
        col = c("blue", "green", "red"),
        ylim = c(0, 1),
        las = 2)  # Rotate x-axis labels

# Add accuracy values on top of bars
text(x = 1:3, 
     y = model_comparison$Accuracy + 0.02, 
     labels = round(model_comparison$Accuracy, 3),
     pos = 3)

# Alternative: Horizontal barplot (better for long model names)
cat("\n=== Creating Horizontal Plot ===\n")
par(mar = c(5, 8, 4, 2))  # Increase left margin
barplot(model_comparison$Accuracy, 
        names.arg = model_comparison$Model,
        main = "F1 Classification Model Comparison",
        xlab = "Accuracy", 
        col = c("lightblue", "lightgreen", "lightcoral"),
        xlim = c(0, 1),
        horiz = TRUE)

# Add grid for better readability
grid(nx = NULL, ny = 0, col = "gray", lty = "dotted")

# Reset margins
par(mar = c(5, 4, 4, 2))

# Print best model information
best_model_index <- which.max(model_comparison$Accuracy)
best_model_name_viz <- model_comparison$Model[best_model_index]
best_accuracy_viz <- model_comparison$Accuracy[best_model_index]

cat("\n=== VISUALIZATION RESULTS ===\n")
cat("Best performing model:", best_model_name_viz, "\n")
cat("Best accuracy:", round(best_accuracy_viz, 4), "\n")
cat("Model improvement over random guessing:", round((best_accuracy_viz - 0.33) * 100, 2), "percentage points\n")

# Create a detailed performance table
performance_table <- data.frame(
  Model = model_comparison$Model,
  Accuracy = round(model_comparison$Accuracy, 4),
  Rank = rank(-model_comparison$Accuracy)
)
print(performance_table)

# Save the visualization results
write.csv(performance_table, "f1_model_detailed_results.csv", row.names = FALSE)
cat("Detailed results saved to 'f1_model_detailed_results.csv'\n")