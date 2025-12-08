# Project 01 - Iris EDA Report

Dataset: Iris (from sklearn)
Rows: 150, Columns: 5

Top 5 rows:
|    |   sepal length (cm) |   sepal width (cm) |   petal length (cm) |   petal width (cm) |   target |
|---:|--------------------:|-------------------:|--------------------:|-------------------:|---------:|
|  0 |                 5.1 |                3.5 |                 1.4 |                0.2 |        0 |
|  1 |                 4.9 |                3   |                 1.4 |                0.2 |        0 |
|  2 |                 4.7 |                3.2 |                 1.3 |                0.2 |        0 |
|  3 |                 4.6 |                3.1 |                 1.5 |                0.2 |        0 |
|  4 |                 5   |                3.6 |                 1.4 |                0.2 |        0 |

Summary statistics:
|       |   sepal length (cm) |   sepal width (cm) |   petal length (cm) |   petal width (cm) |     target |
|:------|--------------------:|-------------------:|--------------------:|-------------------:|-----------:|
| count |          150        |         150        |            150      |         150        | 150        |
| mean  |            5.84333  |           3.05733  |              3.758  |           1.19933  |   1        |
| std   |            0.828066 |           0.435866 |              1.7653 |           0.762238 |   0.819232 |
| min   |            4.3      |           2        |              1      |           0.1      |   0        |
| 25%   |            5.1      |           2.8      |              1.6    |           0.3      |   0        |
| 50%   |            5.8      |           3        |              4.35   |           1.3      |   1        |
| 75%   |            6.4      |           3.3      |              5.1    |           1.8      |   2        |
| max   |            7.9      |           4.4      |              6.9    |           2.5      |   2        |

Missing values per column:
|                   |   0 |
|:------------------|----:|
| sepal length (cm) |   0 |
| sepal width (cm)  |   0 |
| petal length (cm) |   0 |
| petal width (cm)  |   0 |
| target            |   0 |

Key observations:
- Dataset is clean with no missing values.
- Petal length and petal width show strong positive correlation.
- Sepal features are less correlated with petal features.
- Species are reasonably separable via petal dimensions (makes classification easy).

Plots saved in project01_iris/plots/
