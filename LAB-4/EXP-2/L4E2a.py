import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("co2.csv")

# Display first records
print(df.head())

# Understand structure
print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())
