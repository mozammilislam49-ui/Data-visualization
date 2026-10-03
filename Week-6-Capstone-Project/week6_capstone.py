# ============================================================
# NSDC VIRTUAL INTERNSHIP
# Virtual Data Science with Python
# WEEK 6 CAPSTONE PROJECT
# Integrative Capstone Project and Evaluation
#
# Trainee: Mozammil Islam
# Project: Titanic Passenger Survival Analysis
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    silhouette_score
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# 2. LOAD TITANIC DATASET
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

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 3. REMOVE EXACT DUPLICATES
# ============================================================

df = df.drop_duplicates().copy()

print(
    "\nDataset Shape After Duplicate Removal:",
    df.shape
)


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Fill Age using median
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

# Fill Embarked using mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)


print(
    "\nMissing Age:",
    df["Age"].isnull().sum()
)

print(
    "Missing Embarked:",
    df["Embarked"].isnull().sum()
)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# Total family size
df["FamilySize"] = (
    df["SibSp"]
    + df["Parch"]
    + 1
)

# Passenger travelling alone
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
# 6. DESCRIPTIVE STATISTICS
# ============================================================

print("\n================ DESCRIPTIVE STATISTICS ================\n")

print(
    df.describe(
        include="all"
    ).T
)


# ============================================================
# 7. SURVIVAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title(
    "Titanic Passenger Survival Distribution"
)

plt.xlabel(
    "Survived (0 = No, 1 = Yes)"
)

plt.ylabel(
    "Passenger Count"
)

plt.tight_layout()

plt.show()


# ============================================================
# 8. SURVIVAL BY SEX
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Sex",
    hue="Survived"
)

plt.title(
    "Survival by Sex"
)

plt.xlabel("Sex")
plt.ylabel("Passenger Count")

plt.tight_layout()

plt.show()


# ============================================================
# 9. SURVIVAL BY PASSENGER CLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Pclass",
    hue="Survived"
)

plt.title(
    "Survival by Passenger Class"
)

plt.xlabel(
    "Passenger Class"
)

plt.ylabel(
    "Passenger Count"
)

plt.tight_layout()

plt.show()


# ============================================================
# 10. AGE DISTRIBUTION BY SURVIVAL
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Survived",
    kde=True,
    element="step"
)

plt.title(
    "Age Distribution by Survival"
)

plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.show()


# ============================================================
# 11. FARE DISTRIBUTION BY SURVIVAL
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Survived",
    y="Fare"
)

plt.title(
    "Fare Distribution by Survival"
)

plt.xlabel(
    "Survival Status"
)

plt.ylabel("Fare")

plt.tight_layout()

plt.show()


# ============================================================
# 12. CORRELATION HEATMAP
# ============================================================

numeric_columns = [

    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone"

]

correlation_matrix = (
    df[numeric_columns]
    .corr()
)


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Titanic Correlation Heatmap"
)

plt.tight_layout()

plt.show()


# ============================================================
# 13. SUPERVISED LEARNING FEATURES
# ============================================================

features = [

    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Embarked"

]

X = df[features].copy()

y = df["Survived"]


print(
    "\n================ SUPERVISED LEARNING ================\n"
)

print(
    "Input Features:"
)

print(features)

print(
    "\nTarget: Survived"
)


# ============================================================
# 14. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "\nTraining Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# ============================================================
# 15. DEFINE FEATURE GROUPS
# ============================================================

numeric_features = [

    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone"

]

categorical_features = [

    "Sex",
    "Embarked"

]


# ============================================================
# 16. NUMERIC PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([

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
# 17. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

])


# ============================================================
# 18. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([

    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )

])


# ============================================================
# 19. LOGISTIC REGRESSION MODEL
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
# 20. CROSS VALIDATION
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
    "\n================ CROSS VALIDATION ================\n"
)


cv_summary = {}


for metric in scoring:

    score = cv_results[
        "test_" + metric
    ].mean()

    cv_summary[
        metric
    ] = score

    print(
        f"{metric.upper()}: {score:.4f}"
    )


# ============================================================
# 21. TRAIN FINAL SUPERVISED MODEL
# ============================================================

model.fit(

    X_train,

    y_train

)

print(
    "\nLogistic Regression trained successfully."
)


# ============================================================
# 22. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 23. MODEL EVALUATION
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
    y_probability
)


print(
    "\n================ TEST RESULTS ================\n"
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
# 24. CLASSIFICATION REPORT
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
# 25. CONFUSION MATRIX
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
# 26. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(

    y_test,

    y_probability

)

plt.title(
    "ROC Curve – Logistic Regression"
)

plt.tight_layout()

plt.show()


# ============================================================
# 27. LOGISTIC REGRESSION COEFFICIENTS
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


coefficient_df = coefficient_df.sort_values(

    "Coefficient",

    ascending=False

)


print(
    "\n================ MODEL COEFFICIENTS ================\n"
)

print(
    coefficient_df
)


# ============================================================
# 28. UNSUPERVISED LEARNING
# ============================================================

print(
    "\n================ K-MEANS CLUSTERING ================\n"
)


cluster_features = [

    "Pclass",
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "Sex"

]


cluster_data = (
    df[cluster_features]
    .copy()
)


# Encode Sex
cluster_data["Sex"] = cluster_data[
    "Sex"
].map({

    "male": 0,

    "female": 1

})


# Handle remaining missing values
cluster_data["Age"] = cluster_data[
    "Age"
].fillna(

    cluster_data[
        "Age"
    ].median()

)


cluster_data["Fare"] = cluster_data[
    "Fare"
].fillna(

    cluster_data[
        "Fare"
    ].median()

)


cluster_data = cluster_data.dropna()


# ============================================================
# 29. STANDARDIZE CLUSTERING DATA
# ============================================================

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(

    cluster_data

)


# ============================================================
# 30. SELECT NUMBER OF CLUSTERS
# ============================================================

K_range = range(
    2,
    9
)

inertias = []

silhouette_scores = []


for k in K_range:

    km = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )

    labels = km.fit_predict(
        cluster_scaled
    )

    inertias.append(
        km.inertia_
    )

    silhouette_scores.append(

        silhouette_score(
            cluster_scaled,
            labels
        )

    )


# ============================================================
# 31. ELBOW METHOD
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(

    list(K_range),

    inertias,

    marker="o"

)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method"
)

plt.tight_layout()

plt.show()


# ============================================================
# 32. SILHOUETTE SCORE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(

    list(K_range),

    silhouette_scores,

    marker="o"

)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score by K"
)

plt.tight_layout()

plt.show()


# ============================================================
# 33. DISPLAY BEST SILHOUETTE VALUE
# ============================================================

best_index = np.argmax(
    silhouette_scores
)

best_k = list(
    K_range
)[best_index]


print(
    "\nBest K Based on Silhouette Score:",
    best_k
)

print(
    "Best Silhouette Score:",
    round(
        silhouette_scores[
            best_index
        ],
        4
    )
)


# ============================================================
# 34. FINAL K-MEANS MODEL
# ============================================================

# Baseline K=3 for reproducible capstone analysis

k = 3


kmeans = KMeans(

    n_clusters=k,

    random_state=42,

    n_init=10

)


cluster_data[
    "Cluster"
] = kmeans.fit_predict(

    cluster_scaled

)


print(
    "\nCluster Sizes:"
)

print(

    cluster_data[
        "Cluster"
    ]
    .value_counts()
    .sort_index()

)


# ============================================================
# 35. PCA VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2
)


pca_data = pca.fit_transform(

    cluster_scaled

)


pca_df = pd.DataFrame(

    pca_data,

    columns=[

        "PC1",

        "PC2"

    ]

)


pca_df[
    "Cluster"
] = cluster_data[
    "Cluster"
].values


plt.figure(figsize=(9, 6))

sns.scatterplot(

    data=pca_df,

    x="PC1",

    y="PC2",

    hue="Cluster"

)

plt.title(
    "Titanic Passenger Clusters – PCA Projection"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.tight_layout()

plt.show()


# ============================================================
# 36. CLUSTER PROFILE
# ============================================================

cluster_profile = (

    cluster_data
    .groupby(
        "Cluster"
    )
    .agg({

        "Pclass": "mean",

        "Age": "mean",

        "Fare": "mean",

        "SibSp": "mean",

        "Parch": "mean",

        "FamilySize": "mean",

        "Sex": "mean",

        "Cluster": "size"

    })

    .rename(

        columns={

            "Cluster":
            "Passenger_Count"

        }

    )

)


cluster_profile[
    "Female_Percentage"
] = (

    cluster_profile[
        "Sex"
    ]

    * 100

)


print(
    "\n================ CLUSTER PROFILE ================\n"
)

print(
    cluster_profile.round(2)
)


# ============================================================
# 37. POST-CLUSTER SURVIVAL ANALYSIS
# ============================================================

# Survived was NOT used to create clusters.

cluster_survival = df.loc[
    cluster_data.index,
    ["Survived"]
].copy()


cluster_survival[
    "Cluster"
] = cluster_data[
    "Cluster"
].values


survival_by_cluster = (

    cluster_survival
    .groupby(
        "Cluster"
    )[
        "Survived"
    ]
    .mean()

)


print(
    "\n================ SURVIVAL BY CLUSTER ================\n"
)

print(
    survival_by_cluster.round(3)
)


# ============================================================
# 38. SAVE SUPERVISED MODEL RESULTS
# ============================================================

model_results = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC"

    ],

    "Score": [

        accuracy,

        precision,

        recall,

        f1,

        roc_auc

    ]

})


model_results.to_csv(

    "capstone_model_results.csv",

    index=False

)


# ============================================================
# 39. SAVE CROSS VALIDATION RESULTS
# ============================================================

cv_output = pd.DataFrame({

    "Metric":

        list(
            cv_summary.keys()
        ),

    "Mean_Score":

        list(
            cv_summary.values()
        )

})


cv_output.to_csv(

    "capstone_cross_validation.csv",

    index=False

)


# ============================================================
# 40. SAVE COEFFICIENTS
# ============================================================

coefficient_df.to_csv(

    "capstone_logistic_coefficients.csv",

    index=False

)


# ============================================================
# 41. SAVE CLUSTER RESULTS
# ============================================================

cluster_data.to_csv(

    "capstone_clustered_passengers.csv",

    index=False

)


cluster_profile.to_csv(

    "capstone_cluster_profile.csv"

)


# ============================================================
# 42. SAVE CLUSTER DIAGNOSTICS
# ============================================================

cluster_diagnostics = pd.DataFrame({

    "K":
        list(
            K_range
        ),

    "Inertia":
        inertias,

    "Silhouette_Score":
        silhouette_scores

})


cluster_diagnostics.to_csv(

    "capstone_cluster_diagnostics.csv",

    index=False

)


# ============================================================
# 43. FINAL CAPSTONE SUMMARY
# ============================================================

print(
    "\n========================================================"
)

print(
    "WEEK 6 INTEGRATIVE CAPSTONE PROJECT COMPLETED"
)

print(
    "========================================================"
)


print(
    "\nPROJECT:"
)

print(
    "Titanic Passenger Survival Analysis"
)


print(
    "\nSUPERVISED MODEL:"
)

print(
    "Logistic Regression"
)


print(
    f"\nAccuracy : {accuracy:.4f}"
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


print(
    "\nUNSUPERVISED MODEL:"
)

print(
    "K-Means Clustering"
)


print(
    "\nFinal Clusters Used:",
    k
)


print(
    "Best K From Silhouette Analysis:",
    best_k
)


print(
    "\nOutput Files Generated:"
)

print(
    "1. capstone_model_results.csv"
)

print(
    "2. capstone_cross_validation.csv"
)

print(
    "3. capstone_logistic_coefficients.csv"
)

print(
    "4. capstone_clustered_passengers.csv"
)

print(
    "5. capstone_cluster_profile.csv"
)

print(
    "6. capstone_cluster_diagnostics.csv"
)
