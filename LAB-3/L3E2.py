import pandas as pd

# Load dataset (MUST be done in every file)
df = pd.read_csv("StudentsPerformance.csv")

# Select numerical columns
numerical_cols = ['math score', 'reading score', 'writing score']

# Compute Quartiles
Q1 = df[numerical_cols].quantile(0.25)
Q2 = df[numerical_cols].quantile(0.50)
Q3 = df[numerical_cols].quantile(0.75)

print("First Quartile (Q1):\n", Q1)
print("\nSecond Quartile (Median/Q2):\n", Q2)
print("\nThird Quartile (Q3):\n", Q3)
