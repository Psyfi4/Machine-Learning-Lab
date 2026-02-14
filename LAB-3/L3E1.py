import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Display first records
print(df.head())

# Identify numerical columns
numerical_cols = ['math score', 'reading score', 'writing score']

print("\nNumerical Columns:", numerical_cols)

# Measures of Central Tendency
print("\nMean:\n", df[numerical_cols].mean())
print("\nMedian:\n", df[numerical_cols].median())
print("\nMode:\n", df[numerical_cols].mode().iloc[0])

# Measures of Dispersion
print("\nMinimum:\n", df[numerical_cols].min())
print("\nMaximum:\n", df[numerical_cols].max())
print("\nSum:\n", df[numerical_cols].sum())
print("\nVariance:\n", df[numerical_cols].var())
print("\nStandard Deviation:\n", df[numerical_cols].std())
