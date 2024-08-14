import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Carregando os dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Função para identificar e remover outliers usando IQR
def remove_outliers_iqr(df):
    df_no_outliers = df.copy()
    for column in df.columns[:-1]:  # Excluindo a coluna 'species'
        Q1 = df_no_outliers[column].quantile(0.25)
        Q3 = df_no_outliers[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_no_outliers = df_no_outliers[(df_no_outliers[column] >= lower_bound) & (df_no_outliers[column] <= upper_bound)]
    return df_no_outliers

# Remover outliers
df_no_outliers = remove_outliers_iqr(df)

# Comparando o tamanho dos DataFrames
print(f"Número de linhas originais: {df.shape[0]}")
print(f"Número de linhas após remoção de outliers: {df_no_outliers.shape[0]}")

# Boxplots para visualizar outliers antes e depois
for column in df.columns[:-1]:  # Excluindo a coluna 'species'
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot - {column} (Original)')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_no_outliers[column])
    plt.title(f'Boxplot - {column} (Sem Outliers)')
    
    plt.show()
