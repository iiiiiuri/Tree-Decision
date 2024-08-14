import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Gráfico de barras para a variável categórica 'species'
plt.figure(figsize=(8, 4))
sns.countplot(x='species', data=df)
plt.title('Gráfico de Barras - Species')
plt.xlabel('Species')
plt.ylabel('Frequência')
plt.show()
