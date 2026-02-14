# ------------------------------------
# Import Libraries
# ------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------
# Load Dataset
# ------------------------------------
df = pd.read_csv("TvMarketing.csv")

# ------------------------------------
# Define Feature and Target
# ------------------------------------
X = df[['TV']]
y = df['Sales']

# ------------------------------------
# Train-Test Split (80:20)
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------
# Train Model
# ------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------
# Predict Values
# ------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------
# Compute RMSE
# ------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ------------------------------------
# Compute R² Score
# ------------------------------------
r2 = r2_score(y_test, y_pred)

# ------------------------------------
# Display Results
# ------------------------------------
print("RMSE (Root Mean Square Error):", rmse)
print("R² Score (Coefficient of Determination):", r2)
