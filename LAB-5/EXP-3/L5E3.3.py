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

#8. CONFUSION MATRIX HEATMAP

cm_nb = confusion_matrix(y_test, nb_pred)
cm_dt = confusion_matrix(y_test, dt_pred)

fig, ax = plt.subplots(1,2, figsize=(10,4))

sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Naive Bayes")

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title("Decision Tree")

plt.show()

#8.
