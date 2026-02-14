import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ad_click.csv")

# Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

corr = df.corr()

sns.heatmap(corr)
plt.title("Feature Correlation Heatmap")
plt.show()
