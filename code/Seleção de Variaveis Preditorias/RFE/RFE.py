from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import pandas as pd

# Carregando os dados Iris
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(data=X, columns=iris.feature_names)

# Criando o modelo de Árvore de Decisão
model = DecisionTreeClassifier()

# Aplicando RFE
rfe = RFE(model, n_features_to_select=2)  # Seleciona as 2 variáveis mais importantes
rfe.fit(X, y)

# Exibindo as variáveis mais importantes
selected_features = df.columns[rfe.support_]
print("Variáveis selecionadas pelo RFE:")
print(selected_features)
