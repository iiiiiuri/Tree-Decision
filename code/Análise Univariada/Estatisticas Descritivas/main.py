from sklearn.datasets import load_iris
import pandas as pd

#GERA AS ESTATISTICAS DESCRITIVAS DO PROJETO.
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print(df.describe())


