# Project 05 - Titanic Dataset (EDA + Classification)
# Run in VS Code: python project05_titanic.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

Path("project05_titanic").mkdir(exist_ok=True)
Path("project05_titanic/plots").mkdir(exist_ok=True)

# Load titanic dataset from seaborn (requires internet the first time)
df = sns.load_dataset('titanic').dropna(subset=['survived'])

# Basic EDA
df.to_csv("project05_titanic/titanic_raw.csv", index=False)
eda_summary = df.describe(include='all')
eda_summary.to_csv("project05_titanic/eda_summary.csv")

# Preprocess: choose a few columns and encode
df2 = df[['survived','pclass','sex','age','fare','embarked']].dropna()
df2['sex'] = df2['sex'].map({'male':0,'female':1})
df2 = pd.get_dummies(df2, columns=['embarked'], drop_first=True)

X = df2.drop('survived', axis=1)
y = df2['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

with open("project05_titanic/report.md", "w") as f:
    f.write("# Project 05 - Titanic EDA & Classification\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion matrix saved as plots/confusion_matrix.png\n")

# Confusion matrix plot
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Titanic")
plt.savefig("project05_titanic/plots/confusion_matrix.png")
plt.close()

# Survival by class plot
plt.figure(figsize=(6,4))
sns.barplot(x='pclass', y='survived', data=df2)
plt.title("Survival Rate by Pclass")
plt.savefig("project05_titanic/plots/survival_by_pclass.png")
plt.close()

print("Project 05 files created under project05_titanic/")
