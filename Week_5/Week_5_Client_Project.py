# -----------------------------
# DAY 7 – EDA CLIENT PROJECT
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\hp\OneDrive\Documents\Data Analysis and Visualization Expertise\Extracted_Files\5-Days-Live-EDA-and-Feature-Engineering-main\Zomato_Dataset\zomato.csv", encoding="latin1")

# -----------------------------
# 2. Basic Info
# -----------------------------
print(df.head())
print(df.info())
print(df.describe(include="all"))

# -----------------------------
# 3. Clean Rating Column
# -----------------------------
df['Aggregate rating'] = df['Aggregate rating'].astype(str)
df['Aggregate rating'] = df['Aggregate rating'].apply(lambda x: x.split('/')[0] if '/' in x else x)
df['Aggregate rating'] = df['Aggregate rating'].replace(['NEW', '-', 'nan'], np.nan)
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')

# -----------------------------
# 4. Clean String Columns
# -----------------------------
df['City'] = df['City'].str.strip()
df['Cuisines'] = df['Cuisines'].str.strip()
df['Restaurant Name'] = df['Restaurant Name'].str.strip()

# -----------------------------
# 5. Handle Missing Values
# -----------------------------
df['Aggregate rating'] = df['Aggregate rating'].fillna(df['Aggregate rating'].median())
df['Cuisines'] = df['Cuisines'].fillna("Unknown")

# -----------------------------
# 6. Frequency Features
# -----------------------------
df['City_count'] = df['City'].map(df['City'].value_counts())
df['Restaurant_count'] = df['Restaurant Name'].map(df['Restaurant Name'].value_counts())
df['Cuisine_count'] = df['Cuisines'].map(df['Cuisines'].value_counts())

# -----------------------------
# 7. Visualizations
# -----------------------------

# Rating Distribution
plt.figure(figsize=(7,4))
sns.histplot(df['Aggregate rating'], bins=20, kde=True)
plt.title("Distribution of Ratings")
plt.show()

# City Distribution
plt.figure(figsize=(10,5))
df['City'].value_counts().head(10).sort_values(ascending=False).plot(kind='bar')
plt.title("Top 10 Cities")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()

# Votes vs Rating Scatter
plt.figure(figsize=(6,4))
sns.scatterplot(x=df['Votes'], y=df['Aggregate rating'])
plt.title("Votes vs Rating")
plt.show()

# Boxplot of Votes
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Votes'])
plt.title("Votes Outlier Detection")
plt.show()

# Correlation Heatmap (numeric only)
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='viridis')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 8. Save Cleaned Data
# -----------------------------
df.to_csv("zomato_cleaned.csv", index=False)
print("EDA Completed. Clean file saved!")
