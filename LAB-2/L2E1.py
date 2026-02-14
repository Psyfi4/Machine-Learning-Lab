import pandas as pd

df = pd.read_csv(r"C:\Users\arman\OneDrive\Desktop\ML Lab\data.csv")

print(df.head())

print(df.isnull())          # Shows missing positions
print(df.notnull())         # Shows valid values

print("\nMissing Values Count:\n", df.isnull().sum())
