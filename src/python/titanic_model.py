"""Titanic survival: load, clean, fit a logistic regression, predict test.csv."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES = ["Pclass", "Sex", "Boy", "Age", "Fare", "FamilySize", "Embarked_Q", "Embarked_S"]

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def load_csv(name):
    df = pd.read_csv(DATA_DIR / name)
    print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean(df, age_median, fare_median, embarked_mode):
    """Apply the same changes to train and test, printing each one."""
    df = df.copy()

    n = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(age_median)
    print(f"- Filled {n} missing Age values with the training median ({age_median})")

    n = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fare_median)
    print(f"- Filled {n} missing Fare values with the training median ({fare_median:.4f})")

    n = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    print(f"- Filled {n} missing Embarked values with the most common port ({embarked_mode})")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("- Encoded Sex as a number: male = 0, female = 1")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print("- Added FamilySize = SibSp + Parch + 1 (the passenger plus family on board)")

    df["Boy"] = df["Name"].str.contains(", Master.", regex=False).astype(int)
    print(f"- Added Boy = 1 if the title in Name is Master (used for boys), else 0: {df['Boy'].sum()} boys")

    df["Embarked_Q"] = (df["Embarked"] == "Q").astype(int)
    df["Embarked_S"] = (df["Embarked"] == "S").astype(int)
    print("- Turned Embarked into two 0/1 columns, Embarked_Q and Embarked_S (C is the baseline)")

    dropped = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Embarked"]
    df = df.drop(columns=dropped)
    print(f"- Dropped columns not used in the model: {', '.join(dropped)}")
    return df


print("=" * 70)
print("Q13: load train.csv")
print("=" * 70)
train = load_csv("train.csv")
print(train.head())

print()
print("=" * 70)
print("Q14: explore and adjust the data")
print("=" * 70)
print("Missing values in train.csv:")
print(train.isna().sum()[train.isna().sum() > 0].to_string())
print(f"\nOverall survival rate: {train['Survived'].mean():.1%}")
print("\nSurvival rate by sex:")
print(train.groupby("Sex")["Survived"].mean().round(3).to_string())
print("\nSurvival rate by class:")
print(train.groupby("Pclass")["Survived"].mean().round(3).to_string())
print("\nCabin is missing for most passengers, so I drop it instead of filling it in.")

# Fill values come from train.csv only and get reused on test.csv
age_median = train["Age"].median()
fare_median = train["Fare"].median()
embarked_mode = train["Embarked"].mode()[0]

print("\nChanges to train.csv:")
train_clean = clean(train, age_median, fare_median, embarked_mode)
X_train = train_clean[FEATURES]
y_train = train_clean["Survived"]
print(f"\nModel data: {X_train.shape[0]} rows, features = {FEATURES}")
print(X_train.head())

print()
print("=" * 70)
print("Q15: fit a logistic regression on train.csv")
print("=" * 70)
model = LogisticRegression(penalty=None, max_iter=1000)
model.fit(X_train, y_train)
coefs = pd.Series(model.coef_[0], index=FEATURES)
print(f"Intercept: {model.intercept_[0]:.4f}")
print("Coefficients (log-odds):")
print(coefs.round(4).to_string())

print()
print("=" * 70)
print("Q16: accuracy on the training set")
print("=" * 70)
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"Training accuracy: {train_acc:.4f} ({train_acc:.1%}) on {len(y_train)} passengers")

print()
print("=" * 70)
print("Q17: load test.csv and predict")
print("=" * 70)
test = load_csv("test.csv")
print("Changes to test.csv (same steps, using the train.csv fill values):")
test_clean = clean(test, age_median, fare_median, embarked_mode)
test_pred = model.predict(test_clean[FEATURES])
predictions = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
print(f"\nPredicted {test_pred.sum()} of {len(test_pred)} test passengers survive ({test_pred.mean():.1%})")
print("First 10 predictions:")
print(predictions.head(10).to_string(index=False))

print()
print("=" * 70)
print("Q18: accuracy on the test set")
print("=" * 70)
print("test.csv has no Survived column (Kaggle keeps those answers private),")
print("so accuracy can't be computed on it directly. Two ways to check it instead:")

# Hold-out: every 5th passenger (PassengerId divisible by 5), so Python and R use the same rows
holdout = train["PassengerId"] % 5 == 0
holdout_model = LogisticRegression(penalty=None, max_iter=1000)
holdout_model.fit(X_train[~holdout], y_train[~holdout])
holdout_acc = accuracy_score(y_train[holdout], holdout_model.predict(X_train[holdout]))
print(f"\n1) Hold-out accuracy: {holdout_acc:.4f} ({holdout_acc:.1%})")
print(f"   Refit the same model on {(~holdout).sum()} train.csv passengers and tested it on the other")
print(f"   {holdout.sum()} (every PassengerId divisible by 5), which the model never saw.")
print("   This is my estimate of test accuracy.")

print()
gender = load_csv("gender_submission.csv")
merged = predictions.merge(gender, on="PassengerId", suffixes=("_pred", "_gender"))
agree = (merged["Survived_pred"] == merged["Survived_gender"]).mean()
print(f"\n2) Match with gender_submission.csv: {agree:.4f} ({agree:.1%})")
print("   gender_submission.csv is Kaggle's example file that guesses every woman survived and")
print("   every man died. It is not the real answers, so this only shows how close the model")
print("   is to that simple guess, not true accuracy.")
