import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# Line Plot (Trend)
plt.plot(df["Age"], df["Marks"])
plt.title("Trend of Age vs Marks")
plt.xlabel("Age")
plt.ylabel("Marks")
plt.show()

# Bar Plot (Categorical Comparison)
df.groupby("City")["Marks"].mean().plot(kind="bar")
plt.title("Average Marks by City")
plt.show()
