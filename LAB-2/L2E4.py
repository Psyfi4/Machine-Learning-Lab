import pandas as pd

df = pd.read_csv("data.csv")

print("Before:\n", df.dtypes)

df["Age"] = df["Age"].astype(int)

print("\nAfter:\n", df.dtypes)
