# ================== IMPORTS ==================
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Base Models
from sklearn.tree import DecisionTreeClassifier

# Ensemble Models
from sklearn.ensemble import BaggingClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ================== LOAD DATA ==================
data = load_breast_cancer()
X = data.data
y = data.target


# ================== TRAIN-TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================== FEATURE SCALING ==================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ================== BAGGING MODEL ==================
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)


# ================== TRAIN ==================
bagging_model.fit(X_train, y_train)


# ================== PREDICT ==================
y_pred = bagging_model.predict(X_test)


# ================== EVALUATION ==================
print("Bagging Results:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
