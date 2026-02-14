import pandas as pd

df = pd.read_csv("data.csv")

print("Before Cleaning:", df.shape)

df_clean = df.dropna()

print("After Cleaning:", df_clean.shape)
