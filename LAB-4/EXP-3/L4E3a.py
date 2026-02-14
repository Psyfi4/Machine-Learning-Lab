import pandas as pd

df = pd.read_csv("ad_click.csv")

print(df.head())
print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())
