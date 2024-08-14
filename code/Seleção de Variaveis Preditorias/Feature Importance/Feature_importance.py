from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Carregando os dados Iris
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(data=X, columns=iris.feature_names)

# Treinando o modelo de Árvore de Decisão
model = DecisionTreeClassifier()
model.fit(X, y)

# Extraindo a importância das variáveis
importance = model.feature_importances_

# Criando um gráfico para visualizar a importância das variáveis
plt.figure(figsize=(8, 6))
plt.barh(df.columns, importance)
plt.xlabel('Importância')
plt.ylabel('Variáveis')
plt.title('Importância das Variáveis - Árvore de Decisão')
plt.show()

# Exibindo a importância das variáveis
for col, imp in zip(df.columns, importance):
    print(f"Variável: {col}, Importância: {imp:.4f}")
