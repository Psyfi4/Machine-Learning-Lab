import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("TvMarketing.csv")

# Define variables
X = df[['TV']]
y = df['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Plot best-fit line
plt.scatter(X_train, y_train)
plt.plot(X_train, model.predict(X_train))
plt.xlabel("TV Budget")
plt.ylabel("Sales")
plt.title("Best Fit Regression Line")
plt.show()
