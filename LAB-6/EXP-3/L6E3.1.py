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

#8. BAR CHART

models = ['KNN', 'SVM']
train_scores = [train_knn, train_svm]
test_scores = [test_knn, test_svm]

x = np.arange(len(models))

plt.bar(x-0.2, train_scores, 0.4, label="Train Accuracy")
plt.bar(x+0.2, test_scores, 0.4, label="Test Accuracy")

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy Comparison")
plt.legend()

plt.show()
