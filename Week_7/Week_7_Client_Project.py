# California Housing Prediction Model (Client ML Project)

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", round(mse, 2))
print("R² Score:", round(r2, 2))

# Save Predictions to CSV
df_results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

df_results.to_csv("house_price_predictions.csv", index=False)

print("\nPrediction file saved as house_price_predictions.csv")
