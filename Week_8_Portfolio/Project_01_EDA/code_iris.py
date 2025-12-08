# Project 01 - Iris Dataset (EDA)
# Run in VS Code: python project01_iris.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pathlib import Path

Path("project01_iris").mkdir(exist_ok=True)
Path("project01_iris/plots").mkdir(exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv("project01_iris/iris_data.csv", index=False)

# Save basic summary
with open("project01_iris/summary.txt", "w") as f:
    f.write("Shape: " + str(df.shape) + "\n\n")
    f.write("Describe:\n")
    f.write(df.describe().to_string())

# Pairplot (may take a moment)
sns.pairplot(df, hue='target', corner=True)
plt.suptitle("Iris Pairplot", y=1.02)
plt.savefig("project01_iris/plots/pairplot.png", bbox_inches='tight')
plt.close()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title("Correlation Heatmap - Iris")
plt.savefig("project01_iris/plots/correlation_heatmap.png", bbox_inches='tight')
plt.close()

# Boxplots for each feature by species
features = df.columns[:-1]
for col in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='target', y=col, data=df)
    plt.title(f"Boxplot of {col} by species")
    plt.savefig(f"project01_iris/plots/box_{col}.png", bbox_inches='tight')
    plt.close()

# Create short markdown report
report = (
"# Project 01 - Iris EDA Report\n\n"
"Dataset: Iris (from sklearn)\n"
"Rows: " + str(df.shape[0]) + ", Columns: " + str(df.shape[1]) + "\n\n"
"Top 5 rows:\n" + df.head().to_markdown() + "\n\n"
"Summary statistics:\n" + df.describe().to_markdown() + "\n\n"
"Missing values per column:\n" + df.isnull().sum().to_markdown() + "\n\n"
"Key observations:\n- Dataset is clean with no missing values.\n"
"- Petal length and petal width show strong positive correlation.\n"
"- Sepal features are less correlated with petal features.\n"
"- Species are reasonably separable via petal dimensions (makes classification easy).\n\n"
"Plots saved in project01_iris/plots/\n"
)
Path("project01_iris/report.md").write_text(report)
print("Project 01 files created under project01_iris/")
