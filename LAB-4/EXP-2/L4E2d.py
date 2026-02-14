import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("co2.csv")

# Engine Size vs CO2
plt.scatter(df['engine_cc'], df['co2_g_km'])
plt.xlabel("Engine CC")
plt.ylabel("CO2 Emission")
plt.title("Engine Size vs CO2")
plt.show()

# Weight vs CO2
plt.scatter(df['vehicle_weight'], df['co2_g_km'])
plt.xlabel("Vehicle Weight")
plt.ylabel("CO2 Emission")
plt.title("Weight vs CO2")
plt.show()
