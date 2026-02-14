import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("co2.csv")

X = df[['engine_cc', 'vehicle_weight']]
y = df['co2_g_km']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Prediction
y_pred = model.predict(X_test)

# Line Chart
plt.plot(y_test.values)
plt.plot(y_pred)
plt.legend(["Actual", "Predicted"])
plt.title("Actual vs Predicted CO2")
plt.show()
