# ============================================================
# NSDC VIRTUAL INTERNSHIP
# Virtual Data Science with Python
# WEEK 2 PROJECT
# Exploratory Data Analysis and Visualization
# Dataset: Titanic Passenger Dataset
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Seaborn theme
sns.set_theme(style="whitegrid")


# ============================================================
# 2. CREATE FOLDER FOR OUTPUT GRAPHS
# ============================================================

output_folder = "titanic_eda_outputs"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Output folder created:", output_folder)


# ============================================================
# 3. LOAD TITANIC DATASET
# ============================================================

# Make sure train.csv is in the same folder as this Python file
df = pd.read_csv("train.csv")

print("\n================ DATASET LOADED ================\n")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n================ DATASET INFORMATION ================\n")

df.info()

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\n================ DESCRIPTIVE STATISTICS ================\n")

print(df.describe().T)


# Categorical variable summary
categorical_columns = ["Sex", "Pclass", "Embarked"]

print("\nCategorical Variable Summary:")

print(
    df[categorical_columns]
    .describe(include="all")
    .T
)


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")

missing = df.isnull().sum().sort_values(ascending=False)

missing_percentage = (
    missing / len(df) * 100
).round(2)

missing_summary = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_percentage
})

print(missing_summary)


# ============================================================
# 7. CREATE WORKING COPY
# ============================================================

eda = df.copy()

print("\nWorking copy created successfully.")


# ============================================================
# 8. DATA PREPARATION
# ============================================================

# ------------------------------------------------------------
# 8.1 Create FamilySize
# ------------------------------------------------------------

eda["FamilySize"] = (
    eda["SibSp"]
    + eda["Parch"]
    + 1
)


# ------------------------------------------------------------
# 8.2 Handle Missing Age
# ------------------------------------------------------------

age_median = eda["Age"].median()

eda["Age"] = eda["Age"].fillna(age_median)


# ------------------------------------------------------------
# 8.3 Handle Missing Embarked
# ------------------------------------------------------------

embarked_mode = eda["Embarked"].mode()[0]

eda["Embarked"] = eda["Embarked"].fillna(
    embarked_mode
)


# ------------------------------------------------------------
# 8.4 Create CabinKnown
# ------------------------------------------------------------

eda["CabinKnown"] = (
    eda["Cabin"]
    .notna()
    .astype(int)
)


print("\n================ DATA AFTER PREPARATION ================\n")

print(
    eda[
        [
            "Age",
            "SibSp",
            "Parch",
            "FamilySize",
            "Embarked",
            "CabinKnown"
        ]
    ].head()
)


print("\nMissing Values After Treatment:")

print(
    eda[
        [
            "Age",
            "Embarked"
        ]
    ]
    .isnull()
    .sum()
)


# ============================================================
# 9. SURVIVAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

ax = sns.countplot(
    data=eda,
    x="Survived"
)

plt.title(
    "Passenger Survival Distribution",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Survived (0 = No, 1 = Yes)"
)

plt.ylabel(
    "Number of Passengers"
)

# Display count on bars
for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/01_survival_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=eda,
    x="Age",
    bins=30,
    kde=True
)

plt.title(
    "Distribution of Passenger Age",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/02_age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. FARE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=eda,
    x="Fare",
    bins=30,
    kde=True
)

plt.title(
    "Distribution of Passenger Fare",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/03_fare_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. SURVIVAL BY GENDER
# ============================================================

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=eda,
    x="Sex",
    hue="Survived"
)

plt.title(
    "Survival by Gender",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.legend(
    title="Survived",
    labels=["No", "Yes"]
)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/04_survival_by_gender.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. SURVIVAL RATE BY PASSENGER CLASS
# ============================================================

survival_by_class = (
    eda.groupby("Pclass")["Survived"]
    .mean()
    .reset_index()
)

survival_by_class["SurvivalRate"] = (
    survival_by_class["Survived"] * 100
)

print(
    "\n================ SURVIVAL RATE BY CLASS ================\n"
)

print(
    survival_by_class[
        ["Pclass", "SurvivalRate"]
    ]
)


plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=survival_by_class,
    x="Pclass",
    y="SurvivalRate"
)

plt.title(
    "Survival Rate by Passenger Class",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")

plt.ylim(0, 100)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%"
    )

plt.tight_layout()

plt.savefig(
    f"{output_folder}/05_survival_rate_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. AGE DISTRIBUTION BY SURVIVAL STATUS
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=eda,
    x="Survived",
    y="Age"
)

plt.title(
    "Age Distribution by Survival Status",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Survived (0 = No, 1 = Yes)"
)

plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/06_age_by_survival.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. FARE DISTRIBUTION BY PASSENGER CLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=eda,
    x="Pclass",
    y="Fare"
)

plt.title(
    "Fare Distribution by Passenger Class",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/07_fare_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. CORRELATION MATRIX
# ============================================================

numeric_columns = [
    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize"
]

correlation_matrix = (
    eda[numeric_columns]
    .corr()
)


print(
    "\n================ CORRELATION MATRIX ================\n"
)

print(correlation_matrix)


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title(
    "Correlation Matrix of Selected Numerical Variables",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/08_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. SURVIVAL RATE BY CLASS AND GENDER
# ============================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=eda,
    x="Pclass",
    y="Survived",
    hue="Sex"
)

plt.title(
    "Survival Rate by Passenger Class and Gender",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Passenger Class")

plt.ylabel(
    "Mean Survival Rate"
)

plt.legend(
    title="Gender"
)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/09_survival_class_gender.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. FAMILY SIZE ANALYSIS
# ============================================================

plt.figure(figsize=(9, 5))

sns.countplot(
    data=eda,
    x="FamilySize"
)

plt.title(
    "Passenger Family Size Distribution",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Family Size")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/10_family_size_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 19. SURVIVAL RATE BY FAMILY SIZE
# ============================================================

family_survival = (
    eda.groupby("FamilySize")["Survived"]
    .mean()
    .reset_index()
)

family_survival["SurvivalRate"] = (
    family_survival["Survived"] * 100
)


print(
    "\n================ SURVIVAL BY FAMILY SIZE ================\n"
)

print(
    family_survival[
        ["FamilySize", "SurvivalRate"]
    ]
)


plt.figure(figsize=(9, 5))

sns.barplot(
    data=family_survival,
    x="FamilySize",
    y="SurvivalRate"
)

plt.title(
    "Survival Rate by Family Size",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Family Size")
plt.ylabel("Survival Rate (%)")

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/11_survival_by_family_size.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. SURVIVAL RATE BY EMBARKED PORT
# ============================================================

embarked_survival = (
    eda.groupby("Embarked")["Survived"]
    .mean()
    .reset_index()
)

embarked_survival["SurvivalRate"] = (
    embarked_survival["Survived"] * 100
)


print(
    "\n================ SURVIVAL BY EMBARKED PORT ================\n"
)

print(embarked_survival)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=embarked_survival,
    x="Embarked",
    y="SurvivalRate"
)

plt.title(
    "Survival Rate by Port of Embarkation",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Port of Embarkation")
plt.ylabel("Survival Rate (%)")

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/12_survival_by_embarked.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. CABIN INFORMATION AND SURVIVAL
# ============================================================

cabin_survival = (
    eda.groupby("CabinKnown")["Survived"]
    .mean()
    .reset_index()
)

cabin_survival["SurvivalRate"] = (
    cabin_survival["Survived"] * 100
)


print(
    "\n================ CABIN INFORMATION VS SURVIVAL ================\n"
)

print(cabin_survival)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=cabin_survival,
    x="CabinKnown",
    y="SurvivalRate"
)

plt.title(
    "Survival Rate by Cabin Information Availability",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Cabin Known (0 = No, 1 = Yes)"
)

plt.ylabel(
    "Survival Rate (%)"
)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/13_cabin_survival.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 22. IMPORTANT AGGREGATED RESULTS
# ============================================================

print(
    "\n================ FINAL EDA FINDINGS ================\n"
)

total_passengers = len(eda)

total_survivors = eda["Survived"].sum()

total_non_survivors = (
    total_passengers - total_survivors
)

overall_survival_rate = (
    eda["Survived"].mean() * 100
)


print("Total Passengers:", total_passengers)

print(
    "Total Survivors:",
    total_survivors
)

print(
    "Total Non-Survivors:",
    total_non_survivors
)

print(
    f"Overall Survival Rate: "
    f"{overall_survival_rate:.2f}%"
)


# ------------------------------------------------------------
# Survival by Gender
# ------------------------------------------------------------

gender_survival = (
    eda.groupby("Sex")["Survived"]
    .agg(["count", "sum", "mean"])
)

gender_survival["Survival Rate %"] = (
    gender_survival["mean"] * 100
)

print("\nSurvival Statistics by Gender:")

print(gender_survival)


# ------------------------------------------------------------
# Survival by Passenger Class
# ------------------------------------------------------------

class_survival = (
    eda.groupby("Pclass")["Survived"]
    .agg(["count", "sum", "mean"])
)

class_survival["Survival Rate %"] = (
    class_survival["mean"] * 100
)

print("\nSurvival Statistics by Passenger Class:")

print(class_survival)


# ------------------------------------------------------------
# Average Age
# ------------------------------------------------------------

print(
    "\nAverage Passenger Age:",
    round(eda["Age"].mean(), 2)
)


# ------------------------------------------------------------
# Median Age
# ------------------------------------------------------------

print(
    "Median Passenger Age:",
    round(eda["Age"].median(), 2)
)


# ------------------------------------------------------------
# Average Fare
# ------------------------------------------------------------

print(
    "Average Passenger Fare:",
    round(eda["Fare"].mean(), 2)
)


# ------------------------------------------------------------
# Maximum Fare
# ------------------------------------------------------------

print(
    "Maximum Passenger Fare:",
    round(eda["Fare"].max(), 2)
)


# ============================================================
# 23. SAVE CLEANED DATASET
# ============================================================

eda.to_csv(
    f"{output_folder}/cleaned_titanic_eda.csv",
    index=False
)

print(
    "\nCleaned dataset successfully saved as:"
)

print(
    f"{output_folder}/cleaned_titanic_eda.csv"
)


# ============================================================
# 24. SAVE SUMMARY TABLES
# ============================================================

missing_summary.to_csv(
    f"{output_folder}/missing_values_summary.csv"
)

survival_by_class.to_csv(
    f"{output_folder}/survival_by_class.csv",
    index=False
)

family_survival.to_csv(
    f"{output_folder}/survival_by_family_size.csv",
    index=False
)

gender_survival.to_csv(
    f"{output_folder}/survival_by_gender.csv"
)


# ============================================================
# 25. FINAL MESSAGE
# ============================================================

print(
    "\n======================================================"
)

print(
    "TITANIC EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY"
)

print(
    "======================================================"
)

print(
    f"\nAll graphs and output files are available inside "
    f"the '{output_folder}' folder."
)
