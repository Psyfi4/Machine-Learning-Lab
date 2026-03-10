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

#8. ROC CURVE

fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)

auc_nb = auc(fpr_nb, tpr_nb)
auc_dt = auc(fpr_dt, tpr_dt)

plt.plot(fpr_nb, tpr_nb, label="Naive Bayes AUC="+str(auc_nb))
plt.plot(fpr_dt, tpr_dt, label="Decision Tree AUC="+str(auc_dt))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.show()
