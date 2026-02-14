import pandas as pd

df = pd.read_csv("TvMarketing.csv")   # keep file in same folder

print(df.head())
print(df.info())
print(df.describe())
