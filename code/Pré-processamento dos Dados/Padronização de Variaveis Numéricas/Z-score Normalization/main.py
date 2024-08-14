from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Padronização (Z-score Normalization)
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Dados Padronizados (Z-score Normalization):")
print(df_standardized.head())
