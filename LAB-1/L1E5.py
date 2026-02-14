import pandas as pd

data = {
    "Name": ["Arman", "Riya", "Karan", "Megha"],
    "Marks": [85, 90, 78, 92]
}

df = pd.DataFrame(data)

print(df)
print("\nData Types:\n", df.dtypes)
