# install.packages("interactions")
# install.packages("readxl")
# install.packages("glmnet")
# install.packages("lmtest")
# install.packages("car")
# install.packages("strucchange")
library(strucchange)
library(car)
library(interactions)
library(readxl)
library(glmnet)
library(lmtest)
filePath <- file.choose()

dataset <- read.csv(filePath)

# Assuming your_data is your dataframe
explanatory_variables <- c("Mother_Education", "Father_Education", "Mother_Work", "Reason_School_Choice", "Commute_Time", "Weekly_Study_Time", "Alcohol_Weekdays", "Alcohol_Weekends", "School", "Housing_Type", "Desire_Graduate_Education","Has_Internet")

# Create all possible interaction terms
interaction_terms <- combn(explanatory_variables, 2, FUN = function(x) paste(x[1], "*", x[2]), simplify = TRUE)

# Create the formula with main effects and all interaction terms
formula_string <- paste("Grade ~", paste(c(explanatory_variables, interaction_terms), collapse = " + "))
interaction_formula <- as.formula(formula_string)

# Fit the initial model
initial_model <- lm(interaction_formula, data = dataset)

# Backward Selection
backward_model <- step(initial_model, direction = "backward", k = log(nrow(dataset)))

# Forward Selection
forward_model <- step(initial_model, direction = "forward", k = log(nrow(dataset)))
#both
stepwise_model <- step(initial_model, direction = "both", k = log(nrow(dataset)), trace = 0)

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
cat("  Adjusted R-squared:", backward_metrics[3], "\n")

cat("Forward Model:\n")
cat("  AIC:", forward_metrics[1], "\n")
cat("  BIC:", forward_metrics[2], "\n")
cat("  Adjusted R-squared:", forward_metrics[3], "\n")

cat("Stepwise Model:\n")
cat("  AIC:", stepwise_metrics[1], "\n")
cat("  BIC:", stepwise_metrics[2], "\n")
cat("  Adjusted R-squared:", stepwise_metrics[3], "\n")

# cat("Lasso Model:\n")
# cat("  Min Cross-validated MSE:", lasso_metrics[1], "\n")
# cat("  Selected Lambda:", lasso_metrics[2], "\n")

##we choose BackWards
summary(backward_model)
residuals <- residuals(backward_model)

fitted_values <- predict(backward_model)

# Diagnostic plots
par(mfrow = c(2, 2))

##assumption check
# 1. Residuals vs Fitted Values Plot
plot(fitted_values, residuals, main = "Residuals vs Fitted", xlab = "Fitted Values", ylab = "Residuals")
plot(stepwise_model)
plot(stepwise_model)

# 2. Normal Q-Q Plot
qqnorm(residuals)
qqline(residuals, col = 2)

# 3. Scale-Location Plot (Square root of standardized residuals)
sqrt_standardized_residuals <- residuals/sqrt(abs(rstandard(backward_model)))
plot(fitted_values, sqrt_standardized_residuals, main = "Scale-Location Plot", xlab = "Fitted Values", ylab = "Square root of Standardized Residuals")

# 4. Residuals vs Leverage Plot
plot(hatvalues(backward_model), residuals, main = "Residuals vs Leverage", xlab = "Leverage", ylab = "Residuals")

# Reset the plotting layout
par(mfrow = c(1, 1))

# Perform statistical tests
shapiro_test <- shapiro.test(residuals)
cat("Shapiro-Wilk Test for Normality:\n")
print(shapiro_test)

bp_test <- bptest(backward_model)
cat("\nBreusch-Pagan Test for Homoscedasticity:\n")
print(bp_test)

chow_result <- sctest(backward_model, test = "Chow")
print(chow_result)

gq_test <- gqtest(backward_model, alternative = "two.sided")
print(bptest)

