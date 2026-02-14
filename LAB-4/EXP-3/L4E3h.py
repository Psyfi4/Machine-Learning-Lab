import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("ad_click.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

# Define Features & Target
X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# -----------------------------
# Visualize Actual vs Predicted
# -----------------------------
plt.plot(y_test.values[:50], label="Actual")
plt.plot(y_pred[:50], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Ad Click")
plt.xlabel("Samples")
plt.ylabel("Click (0 or 1)")
plt.show()
