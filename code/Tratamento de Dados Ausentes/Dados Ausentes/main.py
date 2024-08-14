import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Introduzindo dados ausentes para exemplo
df.loc[0:5, 'sepal length (cm)'] = None

# Verificando dados ausentes
print("Dados ausentes por coluna:")
print(df.isna().sum())

# Imputação com a média (para dados numéricos)
imputer_mean = SimpleImputer(strategy='mean')
df_imputed_mean = df.copy()
df_imputed_mean[df_imputed_mean.columns[:-1]] = imputer_mean.fit_transform(df_imputed_mean[df_imputed_mean.columns[:-1]])

# Imputação com KNN (para dados numéricos)
imputer_knn = KNNImputer(n_neighbors=5)
df_imputed_knn = df.copy()
df_imputed_knn[df_imputed_knn.columns[:-1]] = imputer_knn.fit_transform(df_imputed_knn[df_imputed_knn.columns[:-1]])

# Verificando novamente dados ausentes
print("\nDados ausentes após imputação com média:")
print(df_imputed_mean.isna().sum())

print("\nDados ausentes após imputação com KNN:")
print(df_imputed_knn.isna().sum())
