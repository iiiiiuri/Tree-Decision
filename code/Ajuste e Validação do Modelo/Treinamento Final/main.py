from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Carregando os dados Iris
iris = load_iris()
X = iris.data
y = X[:, 0]  # Usando a primeira característica (comprimento da sépala) como variável alvo
X = np.delete(X, 0, axis=1)  # Remover a primeira coluna para deixar as outras como preditores

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo o modelo
gb_model = GradientBoostingRegressor(random_state=42)

# Definindo a grade de parâmetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Configurando o Grid Search com validação cruzada
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Executando a busca em grade
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

# Realizando a validação cruzada com o melhor modelo
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Calculando as métricas médias
mean_cv_score = np.mean(cv_scores)
mean_cv_rmse = np.sqrt(-mean_cv_score)

print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)
print("Média das pontuações de validação cruzada (neg_mean_squared_error):", mean_cv_score)
print("Média do RMSE de validação cruzada:", mean_cv_rmse)
