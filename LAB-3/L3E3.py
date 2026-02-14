import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Step 2: Select numerical columns
numerical_cols = ['math score', 'reading score', 'writing score']

# Step 3: Correlation
correlation = df[numerical_cols].corr()
print("Correlation Matrix:\n", correlation)

# Step 4: Covariance
covariance = df[numerical_cols].cov()
print("\nCovariance Matrix:\n", covariance)
