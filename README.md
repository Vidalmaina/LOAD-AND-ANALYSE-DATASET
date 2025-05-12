# LOAD-AND-ANALYSE-DATASET
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set seaborn style for better visuals
sns.set(style="whitegrid")

# ---------------------------
# Task 1: Load and Explore the Dataset
# ---------------------------

try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Add target variable
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display first few rows
print("First five rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData types of each column:")
print(df.dtypes)

print("\nMissing values in each column:")
print(df.isnull().sum())

# For demonstration, let's introduce some missing values artificially
# (Comment out if not needed)
# df.loc[0:5, 'sepal length (cm)'] = np.nan

# Handle missing data: drop rows with missing values
df_clean = df.dropna()
print("\nAfter dropping missing values, dataset shape:", df_clean.shape)

# ---------------------------
# Task 2: Basic Data Analysis
# ---------------------------

# Descriptive statistics
print("\nBasic statistics of numerical columns:")
print(df_clean.describe())

# Group by species and compute mean of each numerical feature
group_means = df_clean.groupby('species').mean()
print("\nMean of features grouped by species:")
print(group_means)

# Identify interesting patterns:
# For example, check which species has the largest average petal length
max_petal_length = group_means['petal length (cm)'].idxmax()
print(f"\nSpecies with the largest average petal length: {max_petal_length}")

# ---------------------------
# Task 3: Data Visualization
# ---------------------------

# 1. Line chart: Since the dataset isn't time-series, we can plot mean petal length across species
plt.figure(figsize=(8, 5))
sns.lineplot(x=group_means.index, y=group_means['petal length (cm)'], marker='o')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Petal Length (cm)')
plt.show()

# 2. Bar chart: Compare average sepal length per species
plt.figure(figsize=(8,5))
sns.barplot(x='species', y='sepal length (cm)', data=df_clean)
plt.title('Average Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(8,5))
sns.histplot(df_clean['sepal width (cm)'], bins=15, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot: Petal length vs. Sepal length colored by species
plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df_clean, palette='Set2', s=100)
plt.title('Petal Length vs. Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# ---------------------------
# Observations and Findings
# ---------------------------

# - The petal length varies significantly between species, with 'virginica' having the longest petals on average.
# - Sepal width has a relatively normal distribution, useful for distinguishing species.
# - Visualizations confirm that petal dimensions are key differentiators among species.
# - The dataset is clean after dropping missing values, ensuring reliable analysis.
