# Import required libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Column names as per UCI IRIS dataset
columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

# Load the IRIS Dataset (NO HEADER in iris.data)
data = pd.read_csv("iris.csv")

# Remove empty rows if present
data = data.dropna()

# Features and target
X = data.iloc[:, 1:5]   # skip Id column
y = data.iloc[:, 5]     # correct Species column

# Convert species labels to numeric values
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Feature scaling (recommended for ANN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ANN model
model = MLPClassifier(hidden_layer_sizes=(10, 8), max_iter=2000, random_state=42)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
