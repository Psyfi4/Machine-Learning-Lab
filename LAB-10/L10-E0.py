# Import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


# Column names
columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

# Load dataset
data = pd.read_csv("iris.csv")

# Remove empty rows
data = data.dropna()

# Features and target
X = data.iloc[:, 1:5].values   # skip Id
y = data.iloc[:, 5].values     # correct Species column

# Encode species labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Convert to categorical (for CNN output layer)
y = to_categorical(y)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for CNN (samples, steps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN Model
from tensorflow.keras.layers import Input

model = Sequential()
model.add(Input(shape=(4,1)))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)

# Prediction
y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Evaluation
print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))
