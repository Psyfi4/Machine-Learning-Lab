import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("co2.csv")

# Select correct columns
features = df[['engine_cc', 'vehicle_weight', 'co2_g_km']]

# Correlation
corr = features.corr()
print(corr)

# Heatmap
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()
