from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Carregando os dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Para regressão, vamos prever o comprimento da sépala (por exemplo) com base nas outras características
# Vamos usar o comprimento da sépala como variável alvo e as demais como preditores
y = X[:, 0]  # Usando a primeira característica (comprimento da sépala) como variável alvo
X = np.delete(X, 0, axis=1)  # Remover a primeira coluna para deixar as outras como preditores

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo: Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

# Avaliação do modelo
r2 = r2_score(y_test, gb_predictions)
mae = mean_absolute_error(y_test, gb_predictions)
rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))

print("R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
