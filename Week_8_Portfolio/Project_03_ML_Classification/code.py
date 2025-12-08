# Project 03 - Breast Cancer Classification
# Run in VS Code: python project03_breast_cancer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from pathlib import Path

Path("project03_breast_cancer").mkdir(exist_ok=True)
Path("project03_breast_cancer/plots").mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Metrics
report_text = classification_report(y_test, y_pred, target_names=data.target_names)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

Path("project03_breast_cancer/classification_report.txt").write_text(report_text)
pd.DataFrame(cm, index=data.target_names, columns=data.target_names).to_csv("project03_breast_cancer/confusion_matrix.csv")

# Confusion matrix plot
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix - Breast Cancer")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("project03_breast_cancer/plots/confusion_matrix.png")
plt.close()

# ROC curve
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Breast Cancer")
plt.legend()
plt.savefig("project03_breast_cancer/plots/roc_curve.png")
plt.close()

report = (
"# Project 03 - Breast Cancer Classification Report\n\n"
"Model: RandomForestClassifier (n_estimators=100)\n\n"
"Classification Report:\n\n" + report_text + "\n\n"
"AUC: " + str(round(roc_auc,3)) + "\n\n"
"Plots saved in project03_breast_cancer/plots/\n"
)
Path("project03_breast_cancer/report.md").write_text(report)
print("Project 03 files created under project03_breast_cancer/")
