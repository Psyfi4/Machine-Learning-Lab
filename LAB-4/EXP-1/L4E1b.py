import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset (REQUIRED in every file)
df = pd.read_csv("TvMarketing.csv")

# Step 2: Check structure (optional but good practice)
print(df.head())

# Step 3: Scatter Plot
plt.scatter(df['TV'], df['Sales'])
plt.xlabel("TV Budget")
plt.ylabel("Sales")
plt.title("TV Budget vs Sales")
plt.show()
