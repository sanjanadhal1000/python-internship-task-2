# Project 02 - Gender vs Salary (Hypothesis Testing)
# Run in VS Code: python project02_gender_salary.py

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("project02_gender_salary").mkdir(exist_ok=True)
Path("project02_gender_salary/plots").mkdir(exist_ok=True)

# Synthetic dataset (replace with your CSV if you have one)
np.random.seed(42)
n_male = 120
n_female = 110
male_salary = np.random.normal(loc=52000, scale=8000, size=n_male)
female_salary = np.random.normal(loc=50000, scale=9000, size=n_female)

df = pd.DataFrame({
    "gender": ["Male"]*n_male + ["Female"]*n_female,
    "salary": np.concatenate([male_salary, female_salary])
})
df.to_csv("project02_gender_salary/gender_salary.csv", index=False)

# Descriptive stats
desc = df.groupby('gender')['salary'].describe()
desc.to_csv("project02_gender_salary/desc_stats.csv")

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x='gender', y='salary', data=df)
plt.title("Salary by Gender - Boxplot")
plt.savefig("project02_gender_salary/plots/box_salary_gender.png")
plt.close()

# KDE plot
plt.figure(figsize=(6,4))
sns.kdeplot(df[df.gender=='Male'].salary, label='Male', fill=True)
sns.kdeplot(df[df.gender=='Female'].salary, label='Female', fill=True)
plt.legend()
plt.title("Salary Distribution by Gender")
plt.savefig("project02_gender_salary/plots/kde_salary_gender.png")
plt.close()

# Two-sample t-test (Welch)
male = df[df.gender=='Male'].salary
female = df[df.gender=='Female'].salary
t_stat, p_value = stats.ttest_ind(male, female, equal_var=False)

report = (
"# Project 02 - Gender vs Salary (Hypothesis Testing)\n\n"
"Null Hypothesis H0: mean salary (Male) = mean salary (Female)\n"
"Alternative H1: mean salary (Male) != mean salary (Female)\n\n"
"Sample sizes: Male=" + str(len(male)) + ", Female=" + str(len(female)) + "\n\n"
"Descriptive statistics:\n" + desc.to_markdown() + "\n\n"
"T-test result:\n"
"t-statistic = " + str(round(t_stat,4)) + "\n"
"p-value = " + str(round(p_value,4)) + "\n\n"
"Interpretation:\n- If p-value < 0.05, reject H0.\n"
)
Path("project02_gender_salary/report.md").write_text(report)
print("Project 02 files created under project02_gender_salary/")
