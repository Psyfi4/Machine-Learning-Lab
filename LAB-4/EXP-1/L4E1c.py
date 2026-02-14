from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset (REQUIRED in every file)
df = pd.read_csv("TvMarketing.csv")

# Step 2: Check structure (optional but good practice)
print(df.head())

X = df[['TV']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
