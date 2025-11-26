import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv(r"C:\Users\hp\OneDrive\Documents\Data Analysis and Visualization Expertise\Day_14_Client_Project\zomato_cleaned.csv")

# ------------------------------
# 2. Clean & prepare data
# ------------------------------
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

# Drop missing ratings
df = df[df['Aggregate rating'].notnull()]

# ------------------------------
# 3. Create two groups
# ------------------------------
group_online = df[df['Has Online delivery'] == 1]['Aggregate rating']
group_offline = df[df['Has Online delivery'] == 0]['Aggregate rating']

# ------------------------------
# 4. Two-sample t-test
# ------------------------------
result = ttest_ind(group_online, group_offline, equal_var=False)

t_stat = round(float(result.statistic), 2)
p_val = round(float(result.pvalue), 4)
dfree = round(float(result.df), 2)

print("\n=== T-Test Results ===")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
print(f"Degrees of freedom: {dfree}")

# ------------------------------
# 5. Mean ratings of both groups
# ------------------------------
mean_online = round(group_online.mean(), 2)
mean_offline = round(group_offline.mean(), 2)

print("\n=== Group Means ===")
print(f"Mean Rating (Online Delivery): {mean_online}")
print(f"Mean Rating (No Delivery): {mean_offline}")

# ------------------------------
# 6. Visualization
# ------------------------------
plt.figure(figsize=(7,5))
sns.boxplot(x='Has Online delivery', y='Aggregate rating', data=df)
plt.xticks([0, 1], ['No Delivery', 'Online Delivery'])
plt.title("Aggregate Ratings: Online vs No Delivery")
plt.xlabel("Delivery Option")
plt.ylabel("Aggregate Rating")
plt.show()





