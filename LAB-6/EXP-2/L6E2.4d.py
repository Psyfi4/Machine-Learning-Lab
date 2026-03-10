#1. import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

#2. LOAD DATASET

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Classes:", data.target_names)

#3. TRAIN TEST SPLIT/

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. TRAIN DECISION TREE CLASSIFIER

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

#5. PREDICT TEST DATA

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:,1]

#6. PRECISION RECALL CURVE

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Decision Tree)")

plt.show()
