import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Criando gráficos de dispersão para todas as combinações de variáveis numéricas
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle('Gráficos de Dispersão - Variáveis Numéricas vs Species', y=1.02)
plt.show()
