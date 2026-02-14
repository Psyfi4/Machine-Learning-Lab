import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("ad_click.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients shape:", model.coef_.shape)
