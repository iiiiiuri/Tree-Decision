from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Codificação One-Hot
encoder = OneHotEncoder(sparse_output=False)
species_encoded = encoder.fit_transform(df[['species']])

# Convertendo para DataFrame e juntando ao DataFrame original
species_encoded_df = pd.DataFrame(species_encoded, columns=encoder.get_feature_names_out(['species']))
df_onehot = pd.concat([df.drop(columns=['species']), species_encoded_df], axis=1)

print("Dados após Codificação One-Hot:")
print(df_onehot.head())
