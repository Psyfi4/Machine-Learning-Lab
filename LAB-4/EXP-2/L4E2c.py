import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("co2.csv")

features = df[['engine_cc', 'vehicle_weight', 'co2_g_km']]

features.plot(kind='box')
plt.title("Outlier Detection")
plt.show()
