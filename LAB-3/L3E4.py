import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset (REQUIRED)
df = pd.read_csv("StudentsPerformance.csv")

# Step 2: Select numerical columns
numerical_cols = ['math score', 'reading score', 'writing score']

# Step 3: Histograms
df[numerical_cols].hist(bins=20)
plt.suptitle("Distribution of Scores")
plt.show()

# Step 4: Boxplot
df[numerical_cols].plot(kind='box')
plt.title("Boxplot of Scores")
plt.show()
