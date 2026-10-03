# ============================================================
# NSDC VIRTUAL INTERNSHIP
# Virtual Data Science with Python
# WEEK 4 PROJECT
# Supervised Learning Model Implementation
# Trainee: Mozammil Islam
# Dataset: Titanic Passenger Dataset
# Model: Logistic Regression
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("train.csv")

print("\n================ DATASET LOADED ================\n")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["Survived"].value_counts())


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# FamilySize represents total family members travelling together
df["FamilySize"] = (
    df["SibSp"]
    + df["Parch"]
    + 1
)

# IsAlone = 1 when passenger has no family member aboard
df["IsAlone"] = (
    df["FamilySize"] == 1
).astype(int)


print("\n================ FEATURE ENGINEERING ================\n")

print(
    df[
        [
            "SibSp",
            "Parch",
            "FamilySize",
            "IsAlone"
        ]
    ].head()
)


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone"
]

X = df[features].copy()

y = df["Survived"]


print("\nSelected Features:")

print(features)

print("\nTarget Variable: Survived")


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n================ TRAIN TEST SPLIT ================\n")

print(
    "Training Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# ============================================================
# 6. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize"
]

categorical_features = [
    "Pclass",
    "Sex",
    "IsAlone"
]


# ============================================================
# 7. NUMERICAL PREPROCESSING PIPELINE
# ============================================================

numerical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )

])


# ============================================================
# 8. CATEGORICAL PREPROCESSING PIPELINE
# ============================================================

categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

])


# ============================================================
# 9. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([

    (
        "num",
        numerical_pipeline,
        numerical_features
    ),

    (
        "cat",
        categorical_pipeline,
        categorical_features
    )

])


# ============================================================
# 10. BUILD LOGISTIC REGRESSION PIPELINE
# ============================================================

model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

])


# ============================================================
# 11. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

print(
    "\n================ TRAINING MODEL ================\n"
)

model.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression model trained successfully."
)


# ============================================================
# 12. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {

    "accuracy": "accuracy",

    "precision": "precision",

    "recall": "recall",

    "f1": "f1",

    "roc_auc": "roc_auc"

}


cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)


print(
    "\n================ CROSS VALIDATION RESULTS ================\n"
)


cv_summary = {}

for metric in scoring:

    mean_score = (
        cv_results[
            "test_" + metric
        ].mean()
    )

    cv_summary[metric] = mean_score

    print(
        f"{metric.upper()}: {mean_score:.4f}"
    )


# ============================================================
# 13. MAKE TEST SET PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. TEST SET EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


print(
    "\n================ TEST SET RESULTS ================\n"
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print(
    "\n================ CLASSIFICATION REPORT ================\n"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title(
    "Confusion Matrix – Logistic Regression"
)

plt.tight_layout()

plt.show()


# ============================================================
# 17. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title(
    "ROC Curve – Logistic Regression"
)

plt.tight_layout()

plt.show()


# ============================================================
# 18. FEATURE COEFFICIENT ANALYSIS
# ============================================================

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model
    .named_steps["classifier"]
    .coef_[0]
)


coefficient_df = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": coefficients

})


coefficient_df = (
    coefficient_df
    .sort_values(
        "Coefficient",
        ascending=False
    )
)


print(
    "\n================ MODEL COEFFICIENTS ================\n"
)

print(
    coefficient_df
)


# ============================================================
# 19. ERROR ANALYSIS
# ============================================================

errors = X_test.copy()

errors["Actual"] = (
    y_test.values
)

errors["Predicted"] = (
    y_pred
)

errors["Survival_Probability"] = (
    y_prob
)


incorrect_predictions = errors[
    errors["Actual"]
    != errors["Predicted"]
]


print(
    "\n================ ERROR ANALYSIS ================\n"
)

print(
    "Total Incorrect Predictions:",
    len(incorrect_predictions)
)

print(
    "\nSample Incorrect Predictions:"
)

print(
    incorrect_predictions.head(10)
)


# ============================================================
# 20. OPTIONAL RANDOM FOREST MODEL
# ============================================================

random_forest_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )

])


random_forest_model.fit(
    X_train,
    y_train
)


rf_pred = random_forest_model.predict(
    X_test
)

rf_prob = random_forest_model.predict_proba(
    X_test
)[:, 1]


rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)


print(
    "\n================ RANDOM FOREST RESULTS ================\n"
)

print(
    f"RF Accuracy : {rf_accuracy:.4f}"
)

print(
    f"RF Precision: {rf_precision:.4f}"
)

print(
    f"RF Recall   : {rf_recall:.4f}"
)

print(
    f"RF F1 Score : {rf_f1:.4f}"
)

print(
    f"RF ROC-AUC  : {rf_auc:.4f}"
)


# ============================================================
# 21. MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy,
        rf_accuracy
    ],

    "Precision": [
        precision,
        rf_precision
    ],

    "Recall": [
        recall,
        rf_recall
    ],

    "F1 Score": [
        f1,
        rf_f1
    ],

    "ROC-AUC": [
        roc_auc,
        rf_auc
    ]

})


print(
    "\n================ MODEL COMPARISON ================\n"
)

print(
    comparison.round(4)
)


# ============================================================
# 22. SAVE MODEL RESULTS
# ============================================================

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

coefficient_df.to_csv(
    "logistic_regression_coefficients.csv",
    index=False
)

incorrect_predictions.to_csv(
    "incorrect_predictions.csv",
    index=False
)


cv_dataframe = pd.DataFrame({
    "Metric": list(cv_summary.keys()),
    "Mean_CV_Score": list(cv_summary.values())
})

cv_dataframe.to_csv(
    "cross_validation_results.csv",
    index=False
)


print(
    "\nOutput files saved successfully:"
)

print(
    "1. model_comparison.csv"
)

print(
    "2. logistic_regression_coefficients.csv"
)

print(
    "3. incorrect_predictions.csv"
)

print(
    "4. cross_validation_results.csv"
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print(
    "\n======================================================"
)

print(
    "WEEK 4 SUPERVISED LEARNING COMPLETED SUCCESSFULLY"
)

print(
    "======================================================"
)

print(
    "\nPrimary Model: Logistic Regression"
)

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(
    f"Test Precision: {precision:.4f}"
)

print(
    f"Test Recall: {recall:.4f}"
)

print(
    f"Test F1 Score: {f1:.4f}"
)

print(
    f"Test ROC-AUC: {roc_auc:.4f}"
)
