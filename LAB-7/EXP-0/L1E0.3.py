# ================== IMPORTS ==================
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Base Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Ensemble Models
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
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


# ================== MODELS ==================

# Bagging Classifier
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

# AdaBoost Classifier
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

# Gradient Boosting Classifier
gboost_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)


# ================== TRAIN MODELS ==================
bagging_model.fit(X_train, y_train)
adaboost_model.fit(X_train, y_train)
gboost_model.fit(X_train, y_train)


# ================== PREDICTIONS ==================
y_pred_bagging = bagging_model.predict(X_test)
y_pred_adaboost = adaboost_model.predict(X_test)
y_pred_gboost = gboost_model.predict(X_test)


# ================== EVALUATION ==================
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ================== RESULTS ==================
evaluate_model("Bagging", y_test, y_pred_bagging)
evaluate_model("AdaBoost", y_test, y_pred_adaboost)
evaluate_model("Gradient Boosting", y_test, y_pred_gboost)
