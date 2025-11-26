Exploratory Data Analysis (EDA) – Zomato Dataset

1. Introduction

This report presents an Exploratory Data Analysis (EDA) of the Zomato restaurant dataset consisting of 9,551 rows and several columns including restaurant name, city, cuisines, rating, and votes.

Objectives

Clean the dataset and fix incorrect formats

Handle missing values and duplicates

Explore patterns using descriptive statistics

Visualize distributions and relationships

Generate insights useful for decision-making

2. Data Cleaning
2.1 Fixing Rating Column

The Aggregate rating column contained values like "NEW", "-", and "4.5/5".

Extracted the numeric part

Replaced invalid entries with NaN

Converted to float

Filled missing values with median rating

2.2 Cleaning String Columns

Trimmed spaces in:

City

Cuisines

Restaurant Name

2.3 Handling Missing Values

Cuisines → replaced with "Unknown"

Aggregate rating → filled with median

Verified using sns.heatmap(df.isnull())

2.4 Removing Duplicates

Checked and removed any duplicate entries.

2.5 Feature Engineering

Added:

City_count: number of restaurants in each city

Restaurant_count: frequency of restaurant names

Cuisine_count: frequency of cuisines

3. Exploratory Data Analysis
3.1 Rating Distribution

(Insert histogram plot)

Insights:

Most restaurants have ratings between 3.0 and 4.5

Slight left-skew (more high ratings)

3.2 Top Cities by Restaurant Count

(Insert bar plot)

Insights:

New Delhi alone has the highest number of restaurants

Delhi NCR (New Delhi, Gurgaon, Noida) dominates the dataset

Most restaurants are in urban metro areas

3.3 Votes Distribution & Outliers

(Insert boxplot of votes)

Insights:

Many extreme outliers

A few restaurants have very high vote counts

Most restaurants have very low votes

3.4 Votes vs Rating

(Insert scatterplot)

Insights:

Very weak positive correlation

Higher votes do NOT guarantee higher ratings

Rating depends more on quality than popularity

3.5 Correlation Heatmap

(Insert heatmap)

Insights:

Only a slight correlation between Aggregate rating and Votes

No strong correlation between other numeric columns

4. Key Patterns & Insights

Ratings generally high; most between 3.5–4.5

Delhi NCR has the largest share of restaurants

Votes heavily skewed → only a few restaurants extremely popular

Weak relationship between rating and votes

Many cuisines unique / appearing only once

After cleaning, dataset is ready for ML or feature engineering

5. Conclusion

The Zomato dataset exhibits strong regional patterns (Delhi NCR dominance), generally high ratings, and skewed popularity distribution. Data cleaning significantly improved quality, making it suitable for modeling and deeper analysis.
