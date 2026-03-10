#1. import libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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

#4. train KNN model

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

#5. train SVM model

svm = SVC(kernel='linear', probability=True)

svm.fit(X_train, y_train)

#6. prediction(for extra caution)

knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)

knn_prob = knn.predict_proba(X_test)[:,1]
svm_prob = svm.predict_proba(X_test)[:,1]

#7. accuracy(for extra caution)

train_knn = knn.score(X_train, y_train)
test_knn = knn.score(X_test, y_test)

train_svm = svm.score(X_train, y_train)
test_svm = svm.score(X_test, y_test)

#8. CONFUSION MATRIX HEATMAP

cm_knn = confusion_matrix(y_test, knn_pred)
cm_svm = confusion_matrix(y_test, svm_pred)

fig, ax = plt.subplots(1,2, figsize=(10,4))

sns.heatmap(cm_knn, annot=True, fmt='d', cmap="Blues", ax=ax[0])
ax[0].set_title("KNN Confusion Matrix")

sns.heatmap(cm_svm, annot=True, fmt='d', cmap="Greens", ax=ax[1])
ax[1].set_title("SVM Confusion Matrix")

plt.show()
