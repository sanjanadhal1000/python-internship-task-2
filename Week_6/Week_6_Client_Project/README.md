Statistical A/B Test on Zomato Dataset

    This project performs a statistical comparison between two groups in the Zomato restaurant dataset.

Objective

    To check whether restaurant ratings differ between:

        Restaurants that offer online delivery

        Restaurants that do not offer online delivery

Methods Used

    Two-sample t-test (Welch)

    Data cleaning with Pandas

    Visualization using Matplotlib/Seaborn

    Confidence level = 95%

Hypotheses

    H₀: There is no difference in mean ratings (μ₁ = μ₂)

    H₁: Mean ratings are different (μ₁ ≠ μ₂)

Files

    analysis.py → Python code

    report.pdf → Statistical report with interpretation

    zomato_cleaned.csv → Dataset

How to Run
    python analysis.py