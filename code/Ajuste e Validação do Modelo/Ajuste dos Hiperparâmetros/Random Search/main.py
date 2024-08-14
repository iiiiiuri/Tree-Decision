from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import uniform, randint

# Carregando os dados Iris
iris = load_iris()
X = iris.data
y = X[:, 0]  # Usando a primeira característica (comprimento da sépala) como variável alvo
X = np.delete(X, 0, axis=1)  # Remover a primeira coluna para deixar as outras como preditores

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo o modelo
gb_model = GradientBoostingRegressor(random_state=42)

# Definindo a distribuição dos parâmetros
param_distributions = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10)
}

# Configurando o Randomized Search
random_search = RandomizedSearchCV(estimator=gb_model, param_distributions=param_distributions, n_iter=50, cv=5, scoring='neg_mean_squared_error', verbose=1, random_state=42)

# Executando a busca aleatória
random_search.fit(X_train, y_train)

# Melhor combinação de parâmetros
print("Melhores parâmetros encontrados:")
print(random_search.best_params_)

# Avaliando o modelo com os melhores parâmetros
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Avaliação do modelo
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
