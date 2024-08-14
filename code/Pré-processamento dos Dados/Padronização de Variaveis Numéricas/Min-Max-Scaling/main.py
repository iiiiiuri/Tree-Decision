from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pandas as pd

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Normalização (Min-Max Scaling)
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Dados Normalizados (Min-Max Scaling):")
print(df_normalized.head())
