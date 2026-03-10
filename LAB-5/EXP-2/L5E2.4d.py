#1. import libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

#2. LOAD DATASET

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Target classes:", data.target_names)

#3. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. DECISION TREE CLASSIFIER

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

#5. PREDICT CLASS LABEL

y_pred = dt_model.predict(X_test)

y_prob = dt_model.predict_proba(X_test)[:,1]

#6. PRECISION RECALL CURVE

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Decision Tree)")

plt.show()
