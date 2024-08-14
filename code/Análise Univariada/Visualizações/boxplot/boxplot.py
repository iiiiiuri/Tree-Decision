import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Plotando boxplots para cada variável numérica
for column in df.columns[:-1]:  # Excluindo a coluna 'species'
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='species', y=column, data=df)
    plt.title(f'Boxplot de {column} por Species')
    plt.xlabel('Species')
    plt.ylabel(column)
    plt.show()
