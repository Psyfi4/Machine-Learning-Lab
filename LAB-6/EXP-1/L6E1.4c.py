#1. import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

import seaborn as sns

#2. LOAD THE DATASET

data = load_breast_cancer()

X = data.data
y = data.target

print("Features shape:", X.shape)
print("Target shape:", y.shape)

#3. TRAIN & TEST SET

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#4. TRAIN KNN CLASSIFIERS

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#5. PREDICT CLASS LABELS

y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)[:,1]


#6. CLASSIFICATION REPORT

print(classification_report(y_test, y_pred_knn))
