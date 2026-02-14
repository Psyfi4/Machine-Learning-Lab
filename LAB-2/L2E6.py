import pandas as pd

df = pd.read_csv("data.csv")

df["City"] = df["City"].replace({"Del": "Delhi", "Mum": "Mumbai"})

print(df["City"].unique())
