# Load necessary libraries
library(tidyverse)
library(readr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(caret)
library(lubridate)  # For time parsing
library(forcats)    # For setting factor levels in ggplot
# library(DMwR)     # Uncomment if using SMOTE for class balancing

# Step 1: Load datasets with proper column types
results <- read_csv("results.csv", col_types = cols(
  positionOrder = col_integer(),
  grid = col_integer(),
  laps = col_integer(),
  fastestLap = col_character()
))
drivers <- read_csv("drivers.csv")

# Step 2: Clean invalid fastestLap entries
results <- results %>%
  filter(!is.na(positionOrder), !is.na(grid), !is.na(laps), !is.na(fastestLap)) %>%
  filter(!fastestLap %in% c("--", "\\N", "", "DNF"))

# Step 3: Create target variable: is_winner (1 if positionOrder == 1, else 0)
results <- results %>%
  mutate(is_winner = ifelse(positionOrder == 1, 1, 0))

# Step 4: Select features and convert fastestLap to numeric
model_data <- results %>%
  mutate(fastestLap = ifelse(grepl(":", fastestLap),
                             as.numeric(ms(fastestLap)),
                             as.numeric(fastestLap))) %>%
  select(is_winner, grid, laps, fastestLap) %>%
  drop_na()

# Step 5: Train-test split (70-30)
set.seed(123)
train_index <- createDataPartition(model_data$is_winner, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Step 6: (Optional) Balance data using SMOTE if needed
# install.packages("DMwR")  # Run once if needed
# library(DMwR)
# train_data <- SMOTE(is_winner ~ ., train_data, perc.over = 300)

# Step 7: Train classification model (Decision Tree)
model <- rpart(is_winner ~ grid + laps + fastestLap, data = train_data, method = "class")

# Step 8: Plot decision tree
rpart.plot(model)

# Step 9: Make predictions
predictions <- predict(model, test_data, type = "class")

# Step 10: Evaluate model
conf_matrix <- confusionMatrix(predictions, factor(test_data$is_winner))
print(conf_matrix)

# Step 11: Accuracy
accuracy <- conf_matrix$overall['Accuracy']
cat("Accuracy: ", accuracy, "\n")

# Step 12: View class imbalance (optional debugging)
cat("\nPrediction breakdown:\n")
print(table(Actual = test_data$is_winner, Predicted = predictions))

# Step 13: Create clean plot with all bars visible
plot_data <- data.frame(
  Predicted = factor(predictions, levels = c(0, 1)),
  Actual = factor(test_data$is_winner, levels = c(0, 1))
)

bar_plot <- ggplot(plot_data, aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("red", "blue")) +
  labs(title = "Predicted vs Actual - Win Classification",
       x = "Actual Winner",
       y = "Count") +
  theme_minimal()

# Step 14: Show and save plot
print(bar_plot)
ggsave("prediction_vs_actual.png", bar_plot, width = 6, height = 4)
