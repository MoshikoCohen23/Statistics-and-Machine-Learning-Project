
library(pscl)
library(strucchange)
library(car)
library(interactions)
library(readxl)
library(glmnet)
library(lmtest)
library(caret)
filePath <- file.choose()

dataset <- read.csv(filePath)

set.seed(123)
index <- createDataPartition(dataset$Grade, p = 0.7, list = FALSE)
dataset$Grade <- ifelse(dataset$Grade == "Fail", 0, 1)
train_data <- dataset[index, ]
test_data <- dataset[-index, ]

explanatory_variables <- c("Desire_Graduate_Education", "Time_with_Friends", "Weekly_Study_Time", "School", "Private_Tutoring", "Commute_Time", "Free_Time_After_School", "Mother_Education", "Father_Education")
interaction_terms <- combn(explanatory_variables, 2, FUN = function(x) paste(x[1], "*", x[2]), simplify = TRUE)

formula_string <- paste("Grade ~", paste(c(explanatory_variables, interaction_terms), collapse = " + "))
interaction_formula <- as.formula(formula_string)

# Fit the initial model
initial_model <- glm(interaction_formula, data = train_data, family="binomial")

# Backward Selection
backward_model <- step(initial_model, direction = "backward", k = log(nrow(train_data)))

# Forward Selection
empty_model <- glm(Grade ~ 1, data = train_data, family = "binomial")
forward_model <- step(empty_model,scope=interaction_formula, direction = "forward", k = log(nrow(train_data)))
AIC(forward_model)

#both
stepwise_model <- step(initial_model, direction = "both", k = log(nrow(train_data)), trace = 0)


backward_metrics <- c(
  AIC(backward_model),
  BIC(backward_model),
  summary(backward_model)$adj.r.squared
)

forward_metrics <- c(
  AIC(forward_model),
  BIC(forward_model),
  summary(forward_model)$adj.r.squared
)

stepwise_metrics <- c(
  AIC(stepwise_model),
  BIC(stepwise_model),
  summary(stepwise_model)$adj.r.squared
)


# Print Results
cat("Backward Model:\n")
cat("  AIC:", backward_metrics[1], "\n")
cat("  BIC:", backward_metrics[2], "\n")

cat("Forward Model:\n")
cat("  AIC:", forward_metrics[1], "\n")
cat("  BIC:", forward_metrics[2], "\n")

cat("Stepwise Model:\n")
cat("  AIC:", stepwise_metrics[1], "\n")
cat("  BIC:", stepwise_metrics[2], "\n")

##we choose BackWards
summary(backward_model)


##calculate f1_score
test_data$Grade <- ifelse(test_data$Grade == 0, "Fail", "Pass")

train_data$Grade <- as.factor(train_data$Grade)

pseudoR2 <- pR2(backward_model)
print(pseudoR2)

test_data$Grade <- as.factor(test_data$Grade)
predicted_probs <- predict(backward_model, newdata = test_data, type = "response")

predicted <- as.factor(ifelse(predicted_probs > 0.8, "Pass", "Fail"))

levels(predicted) <- levels(test_data$Grade)

print(levels(predicted))
print(levels(test_data$Grade))

# Confusion matrix
conf_matrix <- confusionMatrix(predicted, test_data$Grade)
print(conf_matrix)
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']

print(names(conf_matrix$byClass))

f1_score_manual <- 2 * (precision * recall) / (precision + recall)
print(f1_score_manual)


print(conf_matrix)

summary(test_data$Grade)
summary(predicted)
range(predicted_probs)

