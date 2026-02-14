import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("co2.csv")

X = df[['engine_cc', 'vehicle_weight']]
y = df['co2_g_km']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)
