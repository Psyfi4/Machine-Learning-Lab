#1. import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

#2. LOAD DATASET

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Classes:", data.target_names)

#3. TRAIN AND TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. NAIVE BAYES CLASSIFIER

nb = GaussianNB()
nb.fit(X_train, y_train)

#5. CLASS LABEL

y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:,1]

#6. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
