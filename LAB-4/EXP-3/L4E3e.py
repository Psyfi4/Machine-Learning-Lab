import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ad_click.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression()

scores = cross_val_score(model, X, y, cv=5)

print("K-Fold Scores:", scores)
print("Average Accuracy:", scores.mean())
