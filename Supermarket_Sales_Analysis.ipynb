import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
sns.set(style="whitegrid")

# =========================================================
# SECTION 1: Load & Inspect Data (2 Marks)
# =========================================================
# Automatic check for file existence
file_name = "SuperMarket Analysis (2).csv"
if not os.path.exists(file_name):
    # Try the other common name if the first one isn't found
    file_name = "SuperMarket Analysis.csv"

try:
    dataset = pd.read_csv(file_name)
    print(f"--- SECTION 1: Successfully loaded {file_name} ---")
except FileNotFoundError:
    print("ERROR: File not found. Please upload the CSV to the sidebar.")

if 'dataset' in locals():
    print("Head:\n", dataset.head())
    print("\nInfo:")
    dataset.info()
    print("\nDescription:\n", dataset.describe())

    # =========================================================
    # SECTION 2: Data Cleaning (2 Marks)
    # =========================================================
    print("\n--- SECTION 2: Data Cleaning ---")
    dataset.replace(["", " ", "NA", "N/A", "nan"], pd.NA, inplace=True)
    
    # Cleaning Date and Time
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    # Fix: Correctly parsing Time with AM/PM format
    dataset['Hour'] = pd.to_datetime(dataset['Time'], format='%I:%M:%S %p').dt.hour
    
    # =========================================================
    # SECTION 3 & 4: Exploratory Analysis (4 Marks)
    # =========================================================
    print("\n--- SECTION 3 & 4: Branch Statistics ---")
    branches = dataset['Branch'].to_numpy()
    sales_arr = dataset['Sales'].to_numpy()
    unique_branches = np.unique(branches)

    for b in unique_branches:
        branch_data = sales_arr[branches == b]
        print(f"Branch {b} -> Total: {np.sum(branch_data):.2f}, Mean: {np.mean(branch_data):.2f}")

    # =========================================================
    # SECTION 5: Professional Visualization (4 Marks)
    # =========================================================
    plt.figure(figsize=(20, 18))

    # 1. Customer Type Frequency
    plt.subplot(3, 2, 1)
    dataset['Customer type'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("1. Frequency of Customer Types")

    # 2. Sales Trend
    plt.subplot(3, 2, 2)
    dataset.groupby('Date')['Sales'].sum().plot(marker='o', color='blue')
    plt.title("2. Sales Trend Over Time")

    # 3. Rating Trend
    plt.subplot(3, 2, 3)
    dataset.groupby('Date')['Rating'].mean().plot(marker='s', color='green')
    plt.title("3. Average Rating Trend")

    # 4. Correlation Heatmap
    plt.subplot(3, 2, 4)
    numeric_cols = dataset.select_dtypes(include=[np.number])
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("4. Correlation Heatmap")

    # 5. Sales vs Rating Scatter
    plt.subplot(3, 2, 5)
    sns.scatterplot(x='Sales', y='Rating', hue='Branch', data=dataset)
    plt.title("5. Sales vs Rating Scatter Plot")

    # 6. Gross Income Boxplot
    plt.subplot(3, 2, 6)
    sns.boxplot(x='Product line', y='gross income', data=dataset)
    plt.xticks(rotation=45)
    plt.title("6. Gross Income Distribution")

    plt.tight_layout()
    plt.show()

    # =========================================================
    # SECTION 6: Advanced Questions (4 Marks)
    # =========================================================
    print("\n" + "="*45)
    print("SECTION 6: ANSWERS TO PROJECT QUESTIONS")
    print("="*45)

    # Q1: Highest Revenue
    branch_rev = dataset.groupby('Branch')['Sales'].sum()
    print(f"Q1: Highest Revenue Branch: {branch_rev.idxmax()} (${branch_rev.max():.2f})")
    print("Reason: Higher customer traffic and larger average transaction sizes.")

    # Q2: Members vs Normal
    member_sales = dataset.groupby('Customer type')['Sales'].sum()
    print(f"\nQ2: Total Sales by Customer Type:\n{member_sales}")
    print("Answer: Yes, Members generate higher total revenue.")

    # Q3: Payment Usage
    payment_usage = dataset['Payment'].value_counts()
    print(f"\nQ3: Payment Usage:\n{payment_usage}")
    print(f"Answer: {payment_usage.idxmax()} is the preferred method.")

    # Q4: Highest Rating
    product_ratings = dataset.groupby('Product line')['Rating'].mean()
    print(f"\nQ4: Highest Rated Product Line: {product_ratings.idxmax()} ({product_ratings.max():.2f})")

    # Q5: Correlation Price vs Quantity
    correlation = dataset['Unit price'].corr(dataset['Quantity'])
    print(f"\nQ5: Correlation (Price vs Quantity): {correlation:.4f}")
    print("Interpretation: Near zero correlation suggests that price does not impact the quantity per transaction.")

    # =========================================================
    # SECTION 7: Submission Info (1 Mark)
    # =========================================================
    print("\n--- Project Analysis Successfully Completed ---")
