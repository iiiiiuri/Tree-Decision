from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Codificação Ordinal
encoder = OrdinalEncoder()
df_ordinal = df.copy()
df_ordinal['species'] = encoder.fit_transform(df[['species']])

print("Dados após Codificação Ordinal:")
print(df_ordinal.head())
