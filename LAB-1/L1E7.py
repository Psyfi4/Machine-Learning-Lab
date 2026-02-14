import pandas as pd

df = pd.read_csv("data.csv")

print("\nDataset Info:")
print(df.info())

print("\nRows and Columns:", df.shape)

print("\nMissing Values:\n", df.isnull().sum())
