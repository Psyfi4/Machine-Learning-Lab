import pandas as pd

df = pd.read_csv("data.csv")

df.rename(columns={"Name": "Student_Name", "Marks": "Score"}, inplace=True)

print(df.head())
