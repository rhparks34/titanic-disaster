# Titanic survival: load, clean, fit a logistic regression, predict test.csv.

suppressPackageStartupMessages(library(dplyr))

data_dir <- "../data"
features <- c("Pclass", "Sex", "Boy", "Age", "Fare", "FamilySize", "Embarked_Q", "Embarked_S")
options(width = 120)

header <- function(title) {
  cat("\n", strrep("=", 70), "\n", title, "\n", strrep("=", 70), "\n", sep = "")
}

load_csv <- function(name) {
  df <- read.csv(file.path(data_dir, name), na.strings = c("", "NA"))
  cat(sprintf("Loaded %s: %d rows, %d columns\n", name, nrow(df), ncol(df)))
  df
}

# Apply the same changes to train and test, printing each one
clean <- function(df, age_median, fare_median, embarked_mode) {
  n <- sum(is.na(df$Age))
  df <- df %>% mutate(Age = ifelse(is.na(Age), age_median, Age))
  cat(sprintf("- Filled %d missing Age values with the training median (%s)\n", n, age_median))

  n <- sum(is.na(df$Fare))
  df <- df %>% mutate(Fare = ifelse(is.na(Fare), fare_median, Fare))
  cat(sprintf("- Filled %d missing Fare values with the training median (%.4f)\n", n, fare_median))

  n <- sum(is.na(df$Embarked))
  df <- df %>% mutate(Embarked = ifelse(is.na(Embarked), embarked_mode, Embarked))
  cat(sprintf("- Filled %d missing Embarked values with the most common port (%s)\n", n, embarked_mode))

  df <- df %>% mutate(Sex = ifelse(Sex == "female", 1, 0))
  cat("- Encoded Sex as a number: male = 0, female = 1\n")

  df <- df %>% mutate(FamilySize = SibSp + Parch + 1)
  cat("- Added FamilySize = SibSp + Parch + 1 (the passenger plus family on board)\n")

  df <- df %>% mutate(Boy = as.integer(grepl(", Master.", Name, fixed = TRUE)))
  cat(sprintf("- Added Boy = 1 if the title in Name is Master (used for boys), else 0: %d boys\n", sum(df$Boy)))

  df <- df %>% mutate(Embarked_Q = as.integer(Embarked == "Q"),
                      Embarked_S = as.integer(Embarked == "S"))
  cat("- Turned Embarked into two 0/1 columns, Embarked_Q and Embarked_S (C is the baseline)\n")

  dropped <- c("PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Embarked")
  df <- df %>% select(-all_of(dropped))
  cat(sprintf("- Dropped columns not used in the model: %s\n", paste(dropped, collapse = ", ")))
  df
}

accuracy <- function(actual, predicted) mean(actual == predicted)

header("Q13: load train.csv")
train <- load_csv("train.csv")
print(head(train, 5))

header("Q14: explore and adjust the data")
cat("Missing values in train.csv:\n")
missing <- colSums(is.na(train))
print(missing[missing > 0])
cat(sprintf("\nOverall survival rate: %.1f%%\n", 100 * mean(train$Survived)))
cat("\nSurvival rate by sex:\n")
print(as.data.frame(train %>% group_by(Sex) %>% summarise(survival_rate = round(mean(Survived), 3))), row.names = FALSE)
cat("\nSurvival rate by class:\n")
print(as.data.frame(train %>% group_by(Pclass) %>% summarise(survival_rate = round(mean(Survived), 3))), row.names = FALSE)
cat("\nCabin is missing for most passengers, so I drop it instead of filling it in.\n")

# Fill values come from train.csv only and get reused on test.csv
age_median <- median(train$Age, na.rm = TRUE)
fare_median <- median(train$Fare, na.rm = TRUE)
embarked_mode <- names(which.max(table(train$Embarked)))

cat("\nChanges to train.csv:\n")
train_clean <- clean(train, age_median, fare_median, embarked_mode)
cat(sprintf("\nModel data: %d rows, features = %s\n", nrow(train_clean), paste(features, collapse = ", ")))
print(head(train_clean[, features], 5))

header("Q15: fit a logistic regression on train.csv")
formula <- as.formula(paste("Survived ~", paste(features, collapse = " + ")))
model <- glm(formula, data = train_clean, family = binomial)
cat("Coefficients (log-odds):\n")
print(round(coef(model), 4))

header("Q16: accuracy on the training set")
train_pred <- as.integer(predict(model, train_clean, type = "response") > 0.5)
train_acc <- accuracy(train_clean$Survived, train_pred)
cat(sprintf("Training accuracy: %.4f (%.1f%%) on %d passengers\n", train_acc, 100 * train_acc, nrow(train_clean)))

header("Q17: load test.csv and predict")
test <- load_csv("test.csv")
cat("Changes to test.csv (same steps, using the train.csv fill values):\n")
test_clean <- clean(test, age_median, fare_median, embarked_mode)
test_pred <- as.integer(predict(model, test_clean, type = "response") > 0.5)
predictions <- data.frame(PassengerId = test$PassengerId, Survived = test_pred)
cat(sprintf("\nPredicted %d of %d test passengers survive (%.1f%%)\n",
            sum(test_pred), length(test_pred), 100 * mean(test_pred)))
cat("First 10 predictions:\n")
print(head(predictions, 10), row.names = FALSE)

header("Q18: accuracy on the test set")
cat("test.csv has no Survived column (Kaggle keeps those answers private),\n")
cat("so accuracy can't be computed on it directly. Two ways to check it instead:\n")

# Hold-out: every 5th passenger (PassengerId divisible by 5), so Python and R use the same rows
holdout <- train$PassengerId %% 5 == 0
holdout_model <- glm(formula, data = train_clean[!holdout, ], family = binomial)
holdout_pred <- as.integer(predict(holdout_model, train_clean[holdout, ], type = "response") > 0.5)
holdout_acc <- accuracy(train_clean$Survived[holdout], holdout_pred)
cat(sprintf("\n1) Hold-out accuracy: %.4f (%.1f%%)\n", holdout_acc, 100 * holdout_acc))
cat(sprintf("   Refit the same model on %d train.csv passengers and tested it on the other\n", sum(!holdout)))
cat(sprintf("   %d (every PassengerId divisible by 5), which the model never saw.\n", sum(holdout)))
cat("   This is my estimate of test accuracy.\n")

cat("\n")
gender <- load_csv("gender_submission.csv")
merged <- inner_join(predictions, gender, by = "PassengerId", suffix = c("_pred", "_gender"))
agree <- accuracy(merged$Survived_pred, merged$Survived_gender)
cat(sprintf("\n2) Match with gender_submission.csv: %.4f (%.1f%%)\n", agree, 100 * agree))
cat("   gender_submission.csv is Kaggle's example file that guesses every woman survived and\n")
cat("   every man died. It is not the real answers, so this only shows how close the model\n")
cat("   is to that simple guess, not true accuracy.\n")
