"""Generate sample_customer_data.csv"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

ages = np.random.randint(18, 75, n)
genders = np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
incomes = np.random.normal(55000, 18000, n).clip(15000, 150000).round(-2)
tenures = np.random.randint(1, 120, n)
monthly_charges = np.random.normal(65, 25, n).clip(10, 150).round(2)
total_spend = (monthly_charges * tenures * np.random.uniform(0.8, 1.2, n)).round(2)
purchase_freq = np.random.randint(1, 24, n)
spending_score = np.random.randint(1, 100, n)
categories = np.random.choice(
    ["Electronics", "Clothing", "Food", "Sports", "Home", "Beauty", "Books"], n
)
cities = np.random.choice(
    ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata"], n
)
contracts = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
payment_methods = np.random.choice(
    ["Credit Card", "Debit Card", "UPI", "Net Banking"], n
)
support_calls = np.random.poisson(2, n)
satisfaction = np.random.randint(1, 6, n)
num_products = np.random.randint(1, 8, n)
loyalty_points = (spending_score * purchase_freq * np.random.uniform(0.5, 2, n)).round(0).astype(int)
last_purchase_days = np.random.randint(1, 365, n)

# Churn logic
churn_prob = (
    0.05
    + 0.3 * (contracts == "Month-to-month").astype(float)
    + 0.2 * (satisfaction <= 2).astype(float)
    + 0.15 * (support_calls >= 5).astype(float)
    + 0.1 * (last_purchase_days > 180).astype(float)
    - 0.1 * (tenures > 60).astype(float)
    - 0.05 * (spending_score > 70).astype(float)
).clip(0, 0.95)
churn = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

df = pd.DataFrame({
    "CustomerID": [f"C{str(i).zfill(5)}" for i in range(1, n + 1)],
    "Age": ages,
    "Gender": genders,
    "City": cities,
    "AnnualIncome": incomes.astype(int),
    "Tenure": tenures,
    "Contract": contracts,
    "PaymentMethod": payment_methods,
    "MonthlyCharges": monthly_charges,
    "TotalSpend": total_spend,
    "PurchaseFrequency": purchase_freq,
    "SpendingScore": spending_score,
    "NumProducts": num_products,
    "SupportCalls": support_calls,
    "Satisfaction": satisfaction,
    "LoyaltyPoints": loyalty_points,
    "LastPurchaseDays": last_purchase_days,
    "PreferredCategory": categories,
    "Churn": churn,
})

# Introduce ~5% missing values randomly
for col in ["Age", "AnnualIncome", "MonthlyCharges", "SpendingScore", "Satisfaction"]:
    mask = np.random.rand(n) < 0.05
    df.loc[mask, col] = np.nan

df.to_csv("data/sample_customer_data.csv", index=False)
print(f"Dataset created: {len(df)} rows, {len(df.columns)} columns")
print(df.head())
