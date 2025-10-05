# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# carregar o arquivo
file_path = "used_cars.csv"
df = pd.read_csv(file_path)

# %%
# imprimir as 5 primeiras linhas
df.head()

# %%
# informacoes gerais sobre o df
df.info()

# %%
# Remover o " mi." do final de cada linha da coluna 'milage'
df['milage'] = df['milage'].str[:-4]


# %%
# Substituir a vírgula por nada
df['milage'] = df['milage'].str.replace(',', '')

# %%
# Converter a coluna limpa para o tipo numérico (inteiro)
df['milage'] = df['milage'].astype(int)

# %%
# mesmo para coluna price
df['price'] = df['price'].str[1:].str.replace(',', '').astype(int)

# %%
df.info()

# %%
df['fuel_type']

# %%
# combustivel mais comum
predom_fuel = df['fuel_type'].mode()[0]

# %%
# preencher os campos vazios conforme a variavel predominante
df['fuel_type'].fillna(predom_fuel, inplace=True)

# %%
# check 4009 para fuel_type
df.info()



# %%
# informacoes contidas na coluna
df['accident'].value_counts()

# %%
predom_acc = df['accident'].mode()[0]

# %%
df['accident'] = df['accident'].fillna(predom_acc)


# %%
predom_clean = df['clean_title'].mode()[0]

# %%
df['clean_title'] = df['clean_title'].fillna(predom_clean)

# %%
df.info()

# %%
df['fuel_type'].value_counts()

# %%
val_indesejados = ["–", "not supported"]

# %%
df['fuel_type'] = df['fuel_type'].replace(val_indesejados, predom_fuel)

# %%
df['fuel_type'].value_counts()

# %%
# one-hot-encoding "O One-Hot Encoding transforma essa única coluna 
# em várias novas colunas, uma para cada categoria. Cada nova coluna terá apenas os valores 0 ou 1."
fuel_dummies = pd.get_dummies(df['fuel_type'], prefix='fuel')

# %%
fuel_dummies.head()

# %%
df.info()

# %%
# concatenar as duas tabelas
df = pd.concat([df, fuel_dummies], axis=1)

# %%
# remover a coluna fuel_type
df = df.drop('fuel_type', axis=1)

# %%
df.info()

# %%
df['clean_title'].value_counts()

# %%
cols_codificar = ['transmission', 'accident', 'clean_title', 'brand']

# %%
df = pd.get_dummies(df, columns=cols_codificar, drop_first=True)

# %%
# df simplificado
df_modelo = df.drop(columns=['model', 'engine', 'ext_col', 'int_col'])

# %%
# features X e alvo y
X = df_modelo.drop('price', axis=1)
y = df_modelo['price']

# %%
# colunas que usaremos para prever
X

# %%
# coluna que queremos prever
y

# %%
# 0.80 treino e 0.20 teste 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train

# %%
X_test

# %%
# criar e treinar o modelo de regressao
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# %%
previsoes = modelo.predict(X_test)
mse = mean_squared_error(y_test, previsoes)
rmse = np.sqrt(mse)

# %%
mse

# %%
print(f"O erro médio do nosso modelo (RMSE) é: ${mse:,.2f}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Criar o scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=previsoes)

# Criar a linha de "previsão perfeita" (y=x)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)

# Adicionar títulos e rótulos
plt.title('Preços Reais vs. Previsões do Modelo')
plt.xlabel('Preço Real (y_test)')
plt.ylabel('Preço Previsto (previsoes)')
plt.grid(True)
plt.show()

# %%
# vamos remover os outliers > 10^6
limite_preco = 1500000


# %%
# filtrar pelo limite estabelecido
df_no_outlier = df_modelo[df_modelo['price'] < limite_preco]

# %%
# comparar os dfs
print("numero de carros: antes depois:",len(df_modelo), len(df_no_outlier))

# %%
# repetir o processo de modelagem
X_novo = df_no_outlier.drop('price', axis=1)
y_novo = df_no_outlier['price']

# %%
X_train_novo, X_test_novo, y_train_novo, y_test_novo = train_test_split(X_novo, y_novo, test_size=0.2, random_state=42)

# %%
modelo_novo = LinearRegression()
modelo_novo.fit(X_train_novo, y_train_novo)

# %%
previsoes_novo = modelo_novo.predict(X_test_novo)

# %%
rmse_novo = np.sqrt(mean_squared_error(previsoes_novo, y_test_novo))

# %%
print(f"\nO RMSE do modelo original era: ${rmse:,.2f}")
print(f"O NOVO erro médio (RMSE) sem outliers é: ${rmse_novo:,.2f}")

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_novo, y=previsoes_novo)
plt.plot([min(y_test_novo), max(y_test_novo)], [min(y_test_novo), max(y_test_novo)], color='red', linestyle='--', lw=2)
plt.title('Preços Reais vs. Previsões (SEM OUTLIERS)')
plt.xlabel('Preço Real (y_test_novo)')
plt.ylabel('Preço Previsto (previsoes_novo)')
plt.grid(True)
plt.show()

# %%



