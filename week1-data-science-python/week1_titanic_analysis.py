import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------
# Public Titanic dataset from Seaborn
df = sns.load_dataset("titanic")

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe(include="all"))

# ---------------------------------------------------------
# 2. DATA QUALITY CHECK
# ---------------------------------------------------------
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())

# Visualization 1: missing values
missing = df.isnull().sum()
missing = missing[missing > 0]

plt.figure(figsize=(8, 5))
missing.plot(kind="bar")
plt.title("Missing Values Before Cleaning")
plt.xlabel("Column")
plt.ylabel("Number of Missing Values")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_missing_values.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------
# Fill numerical missing values with median
df["age"] = df["age"].fillna(df["age"].median())
df["fare"] = df["fare"].fillna(df["fare"].median())

# Fill categorical missing values with mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

# Remove duplicate records
df = df.drop_duplicates()

# Check final quality
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)

# ---------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------

# Visualization 2: age distribution
plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=20, edgecolor="black")
plt.title("Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_age_distribution.png"), dpi=200)
plt.close()

# Visualization 3: survival status
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="survived")
plt.title("Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_survival_status.png"), dpi=200)
plt.close()

# Visualization 4: gender vs survival
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="sex", hue="survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_gender_survival.png"), dpi=200)
plt.close()

# Visualization 5: correlation heatmap
numeric_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
correlation = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_correlation_heatmap.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 5. INSIGHTS
# ---------------------------------------------------------
overall_survival = df["survived"].mean() * 100
female_survival = df.loc[df["sex"] == "female", "survived"].mean() * 100
male_survival = df.loc[df["sex"] == "male", "survived"].mean() * 100

print("\n--- Key Insights ---")
print(f"Overall survival rate: {overall_survival:.2f}%")
print(f"Female survival rate: {female_survival:.2f}%")
print(f"Male survival rate: {male_survival:.2f}%")

print("\nSurvival rate by passenger class:")
print((df.groupby("pclass")["survived"].mean() * 100).round(2))

print("\nAnalysis completed successfully.")
print(f"Visualizations saved in: {OUTPUT_DIR}/")
