#1. import libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

#2. load dataset

data = load_breast_cancer()

X = data.data
y = data.target

#3. train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#4. naive bayes model

nb = GaussianNB()

nb.fit(X_train, y_train)

#5. decision tree model

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

#6. predictions(for extra caution)

nb_pred = nb.predict(X_test)
dt_pred = dt.predict(X_test)

nb_prob = nb.predict_proba(X_test)[:,1]
dt_prob = dt.predict_proba(X_test)[:,1]

#7. accuracy(for extra caution)

train_nb = nb.score(X_train, y_train)
test_nb = nb.score(X_test, y_test)

train_dt = dt.score(X_train, y_train)
test_dt = dt.score(X_test, y_test)

print("Naive Bayes Train Accuracy:", train_nb)
print("Naive Bayes Test Accuracy:", test_nb)

print("Decision Tree Train Accuracy:", train_dt)
print("Decision Tree Test Accuracy:", test_dt)
