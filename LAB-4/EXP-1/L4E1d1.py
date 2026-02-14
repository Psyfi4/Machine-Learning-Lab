import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
df = pd.read_csv("TvMarketing.csv")

# Step 2: Define predictor and target
X = df[['TV']]
y = df['Sales']

# Step 3: Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Display parameters
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])
