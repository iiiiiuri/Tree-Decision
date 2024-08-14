from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os dados Iris
iris = load_iris()
X = iris.data
y = X[:, 0]  # Usando a primeira característica (comprimento da sépala) como variável alvo
X = np.delete(X, 0, axis=1)  # Remover a primeira coluna para deixar as outras como preditores

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo e treinando o modelo com os melhores parâmetros encontrados
best_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
best_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = best_model.predict(X_test)

# Calculando as métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Avaliação do Modelo:")
print("R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# Gráfico de Dispersão: Previsões vs. Valores Reais
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', color='red')  # Linha de referência
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Gráfico de Dispersão: Previsões vs. Valores Reais')
plt.grid(True)
plt.show()

# Gráfico de Resíduos
residuals = y_test - y_pred

plt.figure(figsize=(12, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Distribuição dos Resíduos')
plt.grid(True)
plt.show()
