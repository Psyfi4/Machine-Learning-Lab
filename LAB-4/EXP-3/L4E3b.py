import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("ad_click.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Check missing values
print(df.isnull().sum())

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data preprocessing completed.")
