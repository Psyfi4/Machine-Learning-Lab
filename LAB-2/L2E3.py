import pandas as pd

df = pd.read_csv("data.csv")

# Replace numerical missing values
df["Marks"].fillna(df["Marks"].mean(), inplace=True)

# Replace categorical missing values
df["City"].fillna("Unknown", inplace=True)

print(df)
