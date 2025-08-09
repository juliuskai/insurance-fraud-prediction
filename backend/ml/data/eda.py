import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{curr_dir}/simulated_fraud_claims.csv")

# 1. Quick data preview
print(df.head())
print("\nData Types:\n", df.dtypes)

# 2. Class distribution
print("\nFraud Class Distribution:")
print(df["is_fraud"].value_counts(normalize=True))

sns.countplot(x="is_fraud", data=df)
plt.title("Fraud vs Non-Fraud Distribution")
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.ylabel("Count")
plt.show()

# 3. Missing values check
print("\nMissing Values:\n", df.isnull().sum())

# 4. Numeric feature distributions
numeric_cols = ["claim_amount", "days_to_submit", "previous_claims_count", "customer_tenure", "location_risk_score"]
df[numeric_cols].hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.tight_layout()
plt.show()

# 5. Correlation matrix (excluding categorical and ID)
plt.figure(figsize=(10, 6))
corr = df[numeric_cols + ["is_fraud"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

### OPTIONAL


# 6. Boxplot: Claim Amount by Fraud
# These help you visually compare distributions between fraud and non-fraud cases.
plt.figure(figsize=(8, 5))
sns.boxplot(x="is_fraud", y="claim_amount", data=df)
plt.title("Claim Amount by Fraud Label")
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.show()

# 6. Boxplot: Days to Submit by Fraud
plt.figure(figsize=(8, 5))
sns.boxplot(x="is_fraud", y="days_to_submit", data=df)
plt.title("Days to Submit by Fraud Label")
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.show()

# 7. Mean comparison grouped by fraud label
# This shows whether fraudulent claims tend to differ on average.
grouped_means = df.groupby("is_fraud")[["claim_amount", "days_to_submit", "previous_claims_count", "customer_tenure", "location_risk_score"]].mean()
print("\nAverage Values by Fraud Label:\n")
print(grouped_means)

# 8. Count of claim types per fraud class
claim_type_counts = pd.crosstab(df["claim_type"], df["is_fraud"], normalize='index')
print("\nClaim Type Distribution by Fraud Label:\n")
print(claim_type_counts)

# Plot
claim_type_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="Set2")
plt.title("Claim Type Distribution by Fraud Status")
plt.xlabel("Claim Type")
plt.ylabel("Proportion")
plt.legend(["Not Fraud", "Fraud"])
plt.show()


