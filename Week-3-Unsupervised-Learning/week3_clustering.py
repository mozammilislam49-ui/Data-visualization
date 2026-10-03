# ============================================================
# NSDC VIRTUAL INTERNSHIP
# Virtual Data Science with Python
# WEEK 3 PROJECT
# Unsupervised Learning and Clustering Analysis
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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("train.csv")

print("\n================ DATASET LOADED ================\n")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Create FamilySize
df["FamilySize"] = (
    df["SibSp"]
    + df["Parch"]
    + 1
)

print("\nFamilySize Feature Created:")

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
# 4. SELECT FEATURES FOR CLUSTERING
# ============================================================

features = [
    "Pclass",
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "Sex"
]

X = df[features].copy()


print("\n================ SELECTED FEATURES ================\n")

print(X.head())


# ============================================================
# 5. HANDLE MISSING AGE VALUES
# ============================================================

age_median = X["Age"].median()

X["Age"] = X["Age"].fillna(
    age_median
)

print(
    "\nMissing Age Values After Imputation:",
    X["Age"].isnull().sum()
)


# ============================================================
# 6. ENCODE SEX VARIABLE
# ============================================================

X["Sex"] = X["Sex"].map({
    "male": 0,
    "female": 1
})

print("\nEncoded Sex Values:")

print(
    X["Sex"]
    .value_counts(dropna=False)
)


# ============================================================
# 7. DROP ANY REMAINING MISSING VALUES
# ============================================================

X = X.dropna()

print(
    "\nShape After Removing Remaining Missing Values:",
    X.shape
)


# ============================================================
# 8. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(
    "\nFeatures Standardized Successfully."
)


# ============================================================
# 9. ELBOW METHOD
# ============================================================

print("\n================ ELBOW METHOD ================\n")

inertias = []

K_range = range(2, 9)

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(
        model.inertia_
    )

    print(
        f"K = {k}, Inertia = {model.inertia_:.2f}"
    )


# Plot Elbow Method
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
    "Elbow Method for Selecting K"
)

plt.xticks(
    list(K_range)
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 10. SILHOUETTE SCORE
# ============================================================

print(
    "\n================ SILHOUETTE SCORE ================\n"
)

sil_scores = []

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    sil_scores.append(score)

    print(
        f"K = {k}, Silhouette Score = {score:.4f}"
    )


# Plot Silhouette Scores
plt.figure(figsize=(8, 5))

plt.plot(
    list(K_range),
    sil_scores,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score by Number of Clusters"
)

plt.xticks(
    list(K_range)
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 11. DISPLAY BEST SILHOUETTE K
# ============================================================

best_index = np.argmax(
    sil_scores
)

best_k = list(K_range)[
    best_index
]

best_score = sil_scores[
    best_index
]


print(
    "\nBest K Based on Silhouette Score:",
    best_k
)

print(
    "Best Silhouette Score:",
    round(best_score, 4)
)


# ============================================================
# 12. APPLY FINAL K-MEANS MODEL
# ============================================================

# Baseline K = 3 as used in the Week 3 report
k = 3

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

X["Cluster"] = kmeans.fit_predict(
    X_scaled
)


print(
    "\n================ CLUSTER ASSIGNMENTS ================\n"
)

print(
    X["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 13. CLUSTER SIZE ANALYSIS
# ============================================================

cluster_counts = (
    X["Cluster"]
    .value_counts()
    .sort_index()
)


print(
    "\nCluster Sizes:"
)

print(
    cluster_counts
)


plt.figure(figsize=(7, 5))

cluster_counts.plot(
    kind="bar"
)

plt.xlabel(
    "Cluster"
)

plt.ylabel(
    "Number of Passengers"
)

plt.title(
    "Cluster Size Distribution"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. PCA FOR 2D VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(
    X_scaled
)


plot_df = pd.DataFrame(
    X_pca,
    columns=[
        "PC1",
        "PC2"
    ]
)

plot_df["Cluster"] = (
    X["Cluster"]
    .values
)


print(
    "\nExplained Variance Ratio:"
)

print(
    pca.explained_variance_ratio_
)


# ============================================================
# 15. PCA CLUSTER SCATTER PLOT
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="deep",
    s=70
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

plt.legend(
    title="Cluster"
)

plt.tight_layout()

plt.show()


# ============================================================
# 16. CLUSTER PROFILING
# ============================================================

profile = X.groupby(
    "Cluster"
).agg({

    "Pclass": "mean",

    "Age": "mean",

    "Fare": "mean",

    "SibSp": "mean",

    "Parch": "mean",

    "FamilySize": "mean",

    "Sex": "mean",

    "Cluster": "size"

}).rename(
    columns={
        "Cluster": "Passenger_Count"
    }
)


profile[
    "Female_Percentage"
] = (
    profile["Sex"] * 100
)


print(
    "\n================ CLUSTER PROFILE ================\n"
)

print(
    profile.round(2)
)


# ============================================================
# 17. OPTIONAL POST-CLUSTERING SURVIVAL ANALYSIS
# ============================================================

# Important:
# Survived is NOT used to create the clusters.
# It is only examined after clustering.

analysis_df = df.loc[
    X.index,
    ["Survived"]
].copy()


analysis_df[
    "Cluster"
] = (
    X["Cluster"]
    .values
)


survival_by_cluster = (
    analysis_df
    .groupby("Cluster")[
        "Survived"
    ]
    .mean()
    .round(3)
)


print(
    "\n================ SURVIVAL BY CLUSTER ================\n"
)

print(
    survival_by_cluster
)


# ============================================================
# 18. CLUSTER PROFILE VISUALIZATION
# ============================================================

profile_plot = (
    profile[
        [
            "Pclass",
            "Age",
            "Fare",
            "FamilySize"
        ]
    ]
    .reset_index()
)


print(
    "\nCluster Profile Summary:"
)

print(
    profile_plot.round(2)
)


# ============================================================
# 19. SAVE CLUSTERED DATA
# ============================================================

clustered_data = X.copy()

clustered_data.to_csv(
    "titanic_clustered.csv",
    index=False
)

print(
    "\nClustered dataset saved as:"
)

print(
    "titanic_clustered.csv"
)


# ============================================================
# 20. SAVE CLUSTER PROFILE
# ============================================================

profile.to_csv(
    "cluster_profile.csv"
)

print(
    "Cluster profile saved as:"
)

print(
    "cluster_profile.csv"
)


# ============================================================
# 21. SAVE DIAGNOSTIC RESULTS
# ============================================================

diagnostics = pd.DataFrame({

    "K": list(K_range),

    "Inertia": inertias,

    "Silhouette_Score": sil_scores

})

diagnostics.to_csv(
    "clustering_diagnostics.csv",
    index=False
)


print(
    "Clustering diagnostics saved as:"
)

print(
    "clustering_diagnostics.csv"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print(
    "\n======================================================"
)

print(
    "WEEK 3 UNSUPERVISED LEARNING COMPLETED SUCCESSFULLY"
)

print(
    "======================================================"
)

print(
    "\nFinal Number of Clusters:",
    k
)

print(
    "Best K From Silhouette Analysis:",
    best_k
)

print(
    "Total Passengers Used:",
    len(X)
)

print(
    "\nCluster Sizes:"
)

print(
    cluster_counts
)

print(
    "\nCluster Profile:"
)

print(
    profile.round(2)
)
