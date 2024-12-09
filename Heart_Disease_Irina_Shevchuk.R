# Loading the database

# Specify the URL of the dataset in the repository
dataset_url <- "https://raw.githubusercontent.com/Irina-Mikhailovna/HarvardX-PH125.9x-Data-Science-Capstone-Heart-Desease/refs/heads/main/heart_cleveland_upload.csv"
# Loaded the database
data <- read.csv(dataset_url, na.strings = "?", header = TRUE)

# Installation and download of packages

if (!require(tidyverse)) install.packages("tidyverse") # Data tidying
if (!require(ggplot2)) install.packages("ggplot2") # Data Visualization
if (!require(ggcorrplot)) install.packages("ggcorrplot") # Correlation matrix
if (!require(dplyr)) install.packages("dplyr") # Data manipulation
if (!require(rpart)) install.packages("rpart") # Building classification and regression trees
if (!require(tidyr)) install.packages("tidyr") # Create tidy data
if (!require(patchwork)) install.packages("patchwork") # Make plot composition
if (!require(corrplot)) install.packages("corrplot") # Correlation matrix
if (!require(caret)) install.packages("caret") # Complex regression and classification problems
if (!require(rpart)) install.packages("rpart") # Decision Trees
if (!require(randomForest)) install.packages("randomForest") # Random Forest
if (!require(xgboost)) install.packages("xgboost") # Gradient Boosting
if (!require(pROC)) install.packages("pROC") # ROC curves
if (!require(stargazer)) install.packages("stargazer") # creates LATEX code
if (!require(tinytex)) install.packages("tinytex") # 'LaTeX' documents
if (!require(gtsummary)) install.packages("gtsummary") # publication-ready analytical and  tables
if (!require(knitr)) install.packages("knitr") # global settings for R Markdown script

library(tidyverse) # Data tidying
library(ggplot2) # Data Visualization
library(ggcorrplot) # Correlation matrix
library(dplyr) # Data manipulation
library(rpart) # Building classification and regression trees
library(tidyr) # Create tidy data
library(patchwork) # Make plot composition
library(corrplot) # Correlation matrix
library(caret) # Complex regression and classification problems
library(rpart) # Decision Trees
library(randomForest) # Random Forest
library(xgboost) # Gradient Boosting
library(pROC) # ROC curves
library(stargazer) # creates LATEX code
library(tinytex) # 'LaTeX' documents
library(gtsummary) # publication-ready analytical and  tables
library(knitr) # global settings for R Markdown script

# Checked for missing values
sum(is.na(data))

#  Summary of the data set structure
str(data)

# In category type data, change the numeric value to a text value to facilitate data analysis

data_text <- data %>% 
  mutate(sex = if_else(sex == 1, "MALE", "FEMALE"),
         cp = if_else(cp == 0,"TYPICAL ANGINA", if_else(cp == 1, "ATYPICAL ANGINA",
                                                        if_else(cp == 2, "NON-ANGINAL PAIN", "ASYMPTOMATIC"))),
         fbs = if_else(fbs == 1, ">120", "<=120"),
         restecg = if_else(restecg == 0, "NORMAL",
                           if_else(restecg == 1, "ABNORMALITY", "PROBABLE OR DEFINITE")),
         exang = if_else(exang == 1, "YES" ,"NO"),
         slope = if_else(slope == 0, "UPSLOPING",
                         if_else(slope == 1, "FLAT", "DOWNSLOPING")),
         thal = if_else(thal == 0, "NORMAL",
                        if_else(thal == 1, "FIXED DEFECT", "REVERSABLE DEFECT")),
         condition = if_else(condition == 1, "Disease", "No Disease")
  ) %>% 
  mutate_if(is.character, as.factor) %>% dplyr::select(sex, cp, fbs,restecg, exang, slope, ca, thal, condition, everything())

# Summary Data_text
summary(data_text)

# Counted the total number of men and women
total_count_sex <- data_text %>% group_by(sex) %>% summarize(Observations = n())

# Counted men and women have disease
Observations <- data_text %>%
  filter(condition == "Disease") %>%
  group_by(sex) %>%
  summarize(Disease_cases = n())

# Combined data and calculated the percentage
percent_disease <- total_count_sex %>%
  left_join(Observations, by = "sex") %>%
  mutate(Disease_cases = replace_na(Disease_cases, 0), 
         "% disease" = (round((Disease_cases/Observations) * 100,2)))

percent_disease

# The division into training and test sets
set.seed(123)
train_index <-  createDataPartition(data$condition, times = 1, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Summary train_data and test_data
Summary_sets_table <- tibble(
  Set = c("train_data", "test_data"),
  Observations = c(nrow(train_data), nrow(test_data)),
  "Disease cases" = c(sum(train_data$condition), sum(test_data$condition)),
  "% disease" = round(c(sum(train_data$condition)/nrow(train_data)*100, sum(test_data$condition)/nrow(test_data)*100),2)
)

Summary_sets_table %>% knitr::kable()

# Logistic regression > Model all predictors (MAP) > Train data
model_log_MAP <- glm(condition ~ ., data = train_data, family = binomial)
summary(model_log_MAP)

# The confusion matrix: Logistic regression > Model all predictors (MAP)
# Predict probabilities for the test data
predictions_log_MAP <- predict(model_log_MAP, newdata = test_data, type = "response")

# Convert probabilities to binary class predictions
predicted_classes_log_MAP <- ifelse(predictions_log_MAP > 0.5, 1, 0)

# Generate the confusion matrix
confusion_matrix_log_MAP <- table(Predicted = predicted_classes_log_MAP, Actual = test_data$condition)

# Print the confusion matrix
print_confusion_matrix_log_MAP <- paste(
  "The confusion matrix:\n\n",
  paste(capture.output(print(confusion_matrix_log_MAP)), collapse = "\n"), "\n\n",
  sprintf("TN: True Negative (correctly predicted 0)       %d", confusion_matrix_log_MAP[1, 1]), "\n",
  sprintf("FP: False Positive (incorrectly predicted 1)    %d", confusion_matrix_log_MAP[1, 2]), "\n",
  sprintf("FN: False Negative (incorrectly predicted 0)    %d", confusion_matrix_log_MAP[2, 1]), "\n",
  sprintf("TP: True Positive (correctly predicted 1)       %d", confusion_matrix_log_MAP[2, 2]), "\n",
  sep = ""
)

cat(print_confusion_matrix_log_MAP)

# RMSE| Accuracy| Precision| Recall| F1 Score:: Logistic regression > Model all predictors (MAP)
# RMSE 
rmse_log_MAP <- sqrt(mean((test_data$condition - predictions_log_MAP)^2))
# Accuracy
accuracy_log_MAP <- sum(diag(confusion_matrix_log_MAP)) / sum(confusion_matrix_log_MAP)
# Precision
precision_log_MAP <- confusion_matrix_log_MAP[2, 2] / sum(confusion_matrix_log_MAP[2, ])
# Recall
recall_log_MAP <- confusion_matrix_log_MAP[2, 2] / sum(confusion_matrix_log_MAP[, 2])
# F1 Score
f1_score_log_MAP <- 2 * ((precision_log_MAP * recall_log_MAP) / (precision_log_MAP + recall_log_MAP))

# Results table: Logistic regression > Model all predictors (MAP)
results_table <- tibble(
  Model = c("Logistic regression: Model all predictors"),
  RMSE = round(c(rmse_log_MAP),5),
  Accuracy = round(c(accuracy_log_MAP),5),
  Precision = round(c(precision_log_MAP),5),
  Recall = c(recall_log_MAP),
  F1_Score = round(c(f1_score_log_MAP),5)
)
results_table %>% knitr::kable()

# Obtaining coefficients, predictors and p-values
# Obtaining coefficients and p-values
coeffs_log_MAP <- summary(model_log_MAP)$coefficients

# Significant predictors (p-value < 0.05)
significant_predictors_log_MAP <- rownames(coeffs_log_MAP)[coeffs_log_MAP[, 4] < 0.05]
cat("Significant predictors:", paste(significant_predictors_log_MAP, collapse = ", "), "\n")

# Insignificant predictors (p-value > 0.05)
insignificant_predictors_log_MAP <- rownames(coeffs_log_MAP)[coeffs_log_MAP[, 4] > 0.05]
cat("Insignificant predictors:", paste(insignificant_predictors_log_MAP, collapse = ", "), "\n")

# Create train data with significant predictors
train_data_MSP <- train_data %>% select(condition, sex, cp, exang, ca, thal)

# Logistic regression > Model significant predictors (MSP) > Train data 
model_log_MSP <- glm(condition ~ ., data = train_data_MSP, family = binomial)
summary(model_log_MSP)

# The confusion matrix: Logistic regression > Model significant predictors (MSP)
# Predict probabilities for the test data
predictions_log_MSP <- predict(model_log_MSP, newdata = test_data, type = "response")

# Convert probabilities to binary class predictions
predicted_classes_log_MSP <- ifelse(predictions_log_MSP > 0.5, 1, 0)

# Generate the confusion matrix
confusion_matrix_log_MSP <- table(Predicted = predicted_classes_log_MSP, Actual = test_data$condition)

# Print the confusion matrix
print_confusion_matrix_log_MSP <- paste(
  "The confusion matrix:\n\n",
  paste(capture.output(print(confusion_matrix_log_MSP)), collapse = "\n"), "\n\n",
  sprintf("TN: True Negative (correctly predicted 0)       %d", confusion_matrix_log_MSP[1, 1]), "\n",
  sprintf("FP: False Positive (incorrectly predicted 1)    %d", confusion_matrix_log_MSP[1, 2]), "\n",
  sprintf("FN: False Negative (incorrectly predicted 0)    %d", confusion_matrix_log_MSP[2, 1]), "\n",
  sprintf("TP: True Positive (correctly predicted 1)       %d", confusion_matrix_log_MSP[2, 2]), "\n",
  sep = ""
)

cat(print_confusion_matrix_log_MSP)

# RMSE| Accuracy| Precision| Recall| F1 Score: Logistic regression > Model significant predictors (MSP) 
# Accuracy
accuracy_log_MSP <- sum(diag(confusion_matrix_log_MSP)) / sum(confusion_matrix_log_MSP)
# Precision
precision_log_MSP <- confusion_matrix_log_MSP[2, 2] / sum(confusion_matrix_log_MSP[2, ])
# Recall
recall_log_MSP <- confusion_matrix_log_MSP[2, 2] / sum(confusion_matrix_log_MSP[, 2])
# F1 Score
f1_score_log_MSP <- 2 * ((precision_log_MSP * recall_log_MSP) / (precision_log_MSP + recall_log_MSP))
# RMSE 
rmse_log_MSP <- sqrt(mean((test_data$condition - predictions_log_MSP)^2))

# The results table: Logistic regression > Model significant predictors (MSP)
results_table <- tibble(
  Models = c("Logistic regression: Model all predictors","Logistic regression: Model significant predictors"),
  RMSE = round(c(rmse_log_MAP,rmse_log_MSP),5),
  Accuracy = round(c(accuracy_log_MAP,accuracy_log_MSP),5),
  Precision = round(c(precision_log_MAP,precision_log_MSP),5),
  Recall = c(recall_log_MAP,recall_log_MSP ),
  F1_Score = round(c(f1_score_log_MAP,f1_score_log_MSP),5)
)
results_table %>% knitr::kable()

# Decision Trees > Model all predictors (MAP) > Train data
model_tree_MAP <- rpart(condition ~ ., data = train_data, method = "class")
summary(model_tree_MAP)

# Plotted decision tree
rpart.plot::rpart.plot(
  model_tree_MAP, 
  type = 5,             
  extra = 104,          
  under = TRUE,
  tweak = 1.2,
  clip.facs = TRUE,
  box.palette = "GnYlRd", 
  branch.lty = 1,       
)

# Separeted the leaves
leaves <- model_tree_MAP$frame[model_tree_MAP$frame$var == "<leaf>", ]

# The path to the leaves
node_paths <- path.rpart(model_tree_MAP, nodes = as.numeric(rownames(leaves)), pretty = 0)

# Calculating the probability of error
error_rate <- 1 - apply(leaves$yval2[, -1], 1, max)

leaf_table <- tibble(
  "Leaf Number" = rownames(leaves),
  "Features" = node_paths,
  "Observations" = leaves$n,
  "Predicted Class" = if_else(leaves$yval==1,"No Disease","Disease"),
  "Disease Probability" = round(leaves$yval2[, 3] / leaves$n,2),
  "Healthy Probability" = round(leaves$yval2[, 2] / leaves$n,2)
)

leaf_table %>% knitr::kable()

# The confusion matrix: Decision Trees > Model all predictors (MAP)
# Predict probabilities for the test data
predictions_tree_MAP <- predict(model_tree_MAP, newdata = test_data, type = "class")

# Convert predictions to numeric
predictions_tree_MAP_numeric <- as.numeric(as.character(predictions_tree_MAP))

# Create the confusion matrix
confusion_matrix_tree_MAP <- table(Predicted = predictions_tree_MAP, Actual = test_data$condition)

# Print the confusion matrix
print_confusion_matrix_tree_MAP <- paste(
  "The confusion matrix:\n\n",
  paste(capture.output(print(confusion_matrix_tree_MAP)), collapse = "\n"), "\n\n",
  sprintf("TN: True Negative (correctly predicted 0)       %d", confusion_matrix_tree_MAP[1, 1]), "\n",
  sprintf("FP: False Positive (incorrectly predicted 1)    %d", confusion_matrix_tree_MAP[1, 2]), "\n",
  sprintf("FN: False Negative (incorrectly predicted 0)    %d", confusion_matrix_tree_MAP[2, 1]), "\n",
  sprintf("TP: True Positive (correctly predicted 1)       %d", confusion_matrix_tree_MAP[2, 2]), "\n",
  sep = ""
)

cat(print_confusion_matrix_tree_MAP)

# RMSE| Accuracy| Precision| Recall| F1 Score: Decision Trees > Model all predictors (MAP)}
#  Accuracy
accuracy_tree_MAP <- sum(diag(confusion_matrix_tree_MAP)) / sum(confusion_matrix_tree_MAP)
#  Precision 
precision_tree_MAP <- confusion_matrix_tree_MAP["1", "1"] / sum(confusion_matrix_tree_MAP["1", ])
#  Recall 
recall_tree_MAP <- confusion_matrix_tree_MAP["1", "1"] / sum(confusion_matrix_tree_MAP[, "1"])
#  F1 Score
f1_score_tree_MAP <- 2 * ((precision_tree_MAP * recall_tree_MAP) / (precision_tree_MAP + recall_tree_MAP))
# Convert predicted class and condition to numeric
predicted_classes_tree_MAP <- as.numeric(as.character(predict(model_tree_MAP, newdata = test_data, type = "class")))
true_classes_tree_MAP <- as.numeric(as.character(test_data$condition))
#  RMSE 
rmse_tree_MAP <- sqrt(mean((test_data$condition - predictions_tree_MAP_numeric)^2))

# The results table: Decision Trees > Model all predictors (MAP)
results_table <- tibble(
  Models = c("Logistic regression: Model all predictors","Logistic regression: Model significant predictors", "Decision Trees: Model all predictors"),
  RMSE = round(c(rmse_log_MAP, rmse_log_MSP, rmse_tree_MAP),5),
  Accuracy = round(c(accuracy_log_MAP, accuracy_log_MSP, accuracy_tree_MAP),5),
  Precision = round(c(precision_log_MAP, precision_log_MSP, precision_tree_MAP),5),
  Recall = c(recall_log_MAP,recall_log_MSP, recall_tree_MAP),
  F1_Score = round(c(f1_score_log_MAP, f1_score_log_MSP, f1_score_tree_MAP),5)
)
results_table %>% knitr::kable()


