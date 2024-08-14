import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Excluindo a coluna 'species' do cálculo de correlação de Pearson
pearson_corr = df.drop(columns=['species']).corr(method='pearson')
print("Correlação de Pearson:")
print(pearson_corr)

# Mapa de calor da correlação de Pearson
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor - Correlação de Pearson')
plt.show()

# Convertendo 'species' para valores numéricos para correlação de Spearman
df['species_encoded'] = pd.Categorical(df['species']).codes

# Calculando a correlação de Spearman
spearman_corr = df.drop(columns=['species']).corr(method='spearman')
print("\nCorrelação de Spearman:")
print(spearman_corr)

# Mapa de calor da correlação de Spearman
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor - Correlação de Spearman')
plt.show()
