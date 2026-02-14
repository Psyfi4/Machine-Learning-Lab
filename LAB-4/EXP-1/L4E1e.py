# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("TvMarketing.csv")

# Define Feature (TV) and Target (Sales)
X = df[['TV']]
y = df['Sales']

# Split Data (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Sales
y_pred = model.predict(X_test)

# Display Actual vs Predicted
comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted Sales:\n")
print(comparison)
