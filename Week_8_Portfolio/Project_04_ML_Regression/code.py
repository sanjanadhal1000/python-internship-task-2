# Project 04 - California Housing Regression
# Run in VS Code: python project04_california.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

Path("project04_california").mkdir(exist_ok=True)
Path("project04_california/plots").mkdir(exist_ok=True)

data = fetch_california_housing(as_frame=True)
df = data.frame
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Save metrics
with open("project04_california/metrics.txt","w") as f:
    f.write(f"Linear Regression MSE: {mse_lr:.4f}, R2: {r2_lr:.4f}\n")
    f.write(f"Random Forest MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}\n")

# Scatter plot actual vs predicted for RF
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted (RF)")
plt.title("Actual vs Predicted - Random Forest")
plt.savefig("project04_california/plots/actual_vs_pred_rf.png")
plt.close()

# Feature importances
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importances - Random Forest")
plt.savefig("project04_california/plots/feature_importances.png")
plt.close()

report = (
"# Project 04 - California Housing Regression Report\n\n"
"Linear Regression: MSE=" + str(round(mse_lr,4)) + ", R2=" + str(round(r2_lr,4)) + "\n"
"Random Forest: MSE=" + str(round(mse_rf,4)) + ", R2=" + str(round(r2_rf,4)) + "\n\n"
"Top feature importances:\n\n" + feat_imp.to_markdown() + "\n\n"
"Plots saved in project04_california/plots/\n"
)
Path("project04_california/report.md").write_text(report)
print("Project 04 files created under project04_california/")
