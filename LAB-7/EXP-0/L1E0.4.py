# ================== IMPORTS ==================
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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


# ================== DEFINE MODELS ==================

# Bagging
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

# AdaBoost
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

# Gradient Boosting
gboost_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Stacking
stacking_model = StackingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier()),
        ('knn', KNeighborsClassifier()),
        ('lr', LogisticRegression(max_iter=5000))
    ],
    final_estimator=LogisticRegression(max_iter=5000),
    cv=5
)


# ================== TRAIN MODELS ==================
bagging_model.fit(X_train, y_train)
adaboost_model.fit(X_train, y_train)
gboost_model.fit(X_train, y_train)
stacking_model.fit(X_train, y_train)


# ================== PREDICTIONS ==================
y_pred_bagging = bagging_model.predict(X_test)
y_pred_adaboost = adaboost_model.predict(X_test)
y_pred_gboost = gboost_model.predict(X_test)
y_pred_stacking = stacking_model.predict(X_test)


# ================== EVALUATION FUNCTION ==================
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ================== EVALUATE ALL ==================
evaluate_model("Bagging", y_test, y_pred_bagging)
evaluate_model("AdaBoost", y_test, y_pred_adaboost)
evaluate_model("Gradient Boosting", y_test, y_pred_gboost)
evaluate_model("Stacking", y_test, y_pred_stacking)
