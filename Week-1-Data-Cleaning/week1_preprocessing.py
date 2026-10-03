# ============================================================
# NSDC VIRTUAL INTERNSHIP
# Virtual Data Science with Python
# WEEK 1 PROJECT
# Data Acquisition, Cleaning, and Preprocessing
# Trainee: Mozammil Islam
# Dataset: Titanic Passenger Dataset
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("train.csv")

print("\n================ DATASET LOADED ================\n")

print("First 5 rows:")
print(df.head())

print("\nRows and Columns:")
print(df.shape)


# ============================================================
# 3. INITIAL DATA EXPLORATION
# ============================================================

print("\n================ INITIAL DATA EXPLORATION ================\n")

print("Column Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe(include="all").T)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 4. MISSING VALUE REPORT
# ============================================================

missing = df.isnull().sum().sort_values(ascending=False)

missing_pct = (
    missing / len(df) * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_pct
})

print("\n================ MISSING VALUE REPORT ================\n")

print(missing_report)


# ============================================================
# 5. CHECK DUPLICATE RECORDS
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows found:", duplicate_count)

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

print(
    "Dataset shape after removing duplicates:",
    df.shape
)


# ============================================================
# 6. CHECK CATEGORICAL CONSISTENCY
# ============================================================

print("\n================ CATEGORICAL VALUES ================\n")

for col in ["Sex", "Embarked"]:

    if col in df.columns:

        print(f"\nUnique values in {col}:")

        print(
            df[col]
            .dropna()
            .unique()
        )


# ============================================================
# 7. STANDARDIZE CATEGORICAL VALUES
# ============================================================

df["Sex"] = (
    df["Sex"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["Embarked"] = (
    df["Embarked"]
    .astype("string")
    .str.strip()
    .str.upper()
)


print("\nSex Values After Standardization:")
print(
    df["Sex"]
    .value_counts(dropna=False)
)

print("\nEmbarked Values After Standardization:")
print(
    df["Embarked"]
    .value_counts(dropna=False)
)


# ============================================================
# 8. HANDLE MISSING AGE VALUES
# ============================================================

age_median = df["Age"].median()

print("\nMedian Age:", age_median)

df["Age"] = (
    df["Age"]
    .fillna(age_median)
)

print(
    "Remaining Missing Age Values:",
    df["Age"].isnull().sum()
)


# ============================================================
# 9. HANDLE MISSING EMBARKED VALUES
# ============================================================

embarked_mode = (
    df["Embarked"]
    .mode()[0]
)

print(
    "\nMost Frequent Embarked Category:",
    embarked_mode
)

df["Embarked"] = (
    df["Embarked"]
    .fillna(embarked_mode)
)

print(
    "Remaining Missing Embarked Values:",
    df["Embarked"].isnull().sum()
)


# ============================================================
# 10. HANDLE CABIN INFORMATION
# ============================================================

# Instead of filling Cabin with artificial values,
# create a binary variable showing whether Cabin is known.

df["CabinKnown"] = (
    df["Cabin"]
    .notna()
    .astype(int)
)

print("\nCabin Information Availability:")

print(
    df["CabinKnown"]
    .value_counts()
)


# ============================================================
# 11. FARE OUTLIER VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Fare"]
)

plt.title(
    "Fare Distribution – Outlier Inspection"
)

plt.xlabel("Fare")

plt.tight_layout()

plt.show()


# ============================================================
# 12. DETECT FARE OUTLIERS USING IQR METHOD
# ============================================================

Q1 = df["Fare"].quantile(0.25)

Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


fare_outliers = df[
    (df["Fare"] < lower_bound)
    |
    (df["Fare"] > upper_bound)
]


print("\n================ FARE OUTLIER ANALYSIS ================\n")

print("Q1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print(
    "Lower Bound:",
    lower_bound
)

print(
    "Upper Bound:",
    upper_bound
)

print(
    "Potential Fare Outliers:",
    len(fare_outliers)
)


# ============================================================
# 13. LOG TRANSFORMATION OF FARE
# ============================================================

df["Fare_log"] = np.log1p(
    df["Fare"]
)

print("\nFare Log Transformation Created.")

print(
    df[
        ["Fare", "Fare_log"]
    ].head()
)


# ============================================================
# 14. CHECK ERRONEOUS OR IMPOSSIBLE VALUES
# ============================================================

checks = {

    "negative_age":
        (df["Age"] < 0).sum(),

    "negative_fare":
        (df["Fare"] < 0).sum(),

    "invalid_survived":
        (~df["Survived"].isin([0, 1])).sum(),

    "invalid_pclass":
        (~df["Pclass"].isin([1, 2, 3])).sum()

}


print("\n================ DATA VALIDATION ================\n")

for name, count in checks.items():

    print(
        name,
        ":",
        count
    )


# ============================================================
# 15. FEATURE ENGINEERING - FAMILY SIZE
# ============================================================

df["FamilySize"] = (
    df["SibSp"]
    +
    df["Parch"]
    +
    1
)

print("\nFamily Size Feature:")

print(
    df[
        [
            "SibSp",
            "Parch",
            "FamilySize"
        ]
    ].head()
)


# ============================================================
# 16. DROP RAW CABIN COLUMN
# ============================================================

df = df.drop(
    columns=["Cabin"]
)

print(
    "\nCabin column removed."
)


# ============================================================
# 17. BINARY ENCODING OF SEX
# ============================================================

df["Sex"] = df["Sex"].map({

    "male": 0,

    "female": 1

})

print("\nEncoded Sex Values:")

print(
    df["Sex"]
    .value_counts(dropna=False)
)


# ============================================================
# 18. ONE-HOT ENCODING OF EMBARKED
# ============================================================

df = pd.get_dummies(

    df,

    columns=["Embarked"],

    drop_first=True,

    dtype=int
)


print(
    "\nEmbarked successfully one-hot encoded."
)


# ============================================================
# 19. FINAL DATASET VALIDATION
# ============================================================

print("\n================ FINAL DATASET VALIDATION ================\n")

print(
    "Final Dataset Shape:",
    df.shape
)

print("\nRemaining Missing Values:")

print(
    df.isnull().sum()
)

print("\nFinal Data Types:")

print(
    df.dtypes
)

print("\nFirst 5 Rows of Cleaned Dataset:")

print(
    df.head()
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n================ CLEANING SUMMARY ================\n")

print(
    "Total Rows:",
    len(df)
)

print(
    "Duplicate Rows Remaining:",
    df.duplicated().sum()
)

print(
    "Missing Age:",
    df["Age"].isnull().sum()
)

print(
    "Potential Fare Outliers Identified:",
    len(fare_outliers)
)

print(
    "FamilySize Feature Created: Yes"
)

print(
    "CabinKnown Feature Created: Yes"
)

print(
    "Fare_log Feature Created: Yes"
)


# ============================================================
# 21. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "titanic_cleaned.csv",
    index=False
)

print(
    "\nCleaned dataset saved successfully as:"
)

print(
    "titanic_cleaned.csv"
)


# ============================================================
# 22. SAVE MISSING VALUE REPORT
# ============================================================

missing_report.to_csv(
    "missing_values_report.csv"
)

print(
    "Missing value report saved as:"
)

print(
    "missing_values_report.csv"
)


# ============================================================
# 23. PROJECT COMPLETION MESSAGE
# ============================================================

print(
    "\n======================================================"
)

print(
    "WEEK 1 DATA CLEANING AND PREPROCESSING COMPLETED"
)

print(
    "======================================================"
)
