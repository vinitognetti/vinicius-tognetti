# Carregando as bibliotecas necessárias
import numpy as np # Cálculo vetorizado
import pandas as pd # Manipulação de dados
import matplotlib.pyplot as plt # Gráficos
import seaborn as sns # Gráficos mais bonitos
import statsmodels.api as sm # Modelos estatísticos

# ------------------------+
# (1) Carregando os dados |
# ------------------------+

# Definindo o caminho dos dados
caminho = '/content/Ecommerce_DBS.csv'

# Carregando os dados em um dataframe pandas
dados = pd.read_csv(caminho)

# Transformando o tipo de dados da coluna de data de compra
dados['Purchase Date'] = pd.to_datetime(dados['Purchase Date'], format='%d/%m/%Y')

# Transformando o tipo de dados das colunas de (1) Categoria do Produto,
# (2) Gênero, (3) Meio, (4) País e (5) Estado em categóricas
for col in ['Product Category', 'Gender', 'Source', 'Country', 'State']:

  dados[col] = pd.Categorical(dados[col])

# ----------------------------------------------+
# (2) Produtos mais vendidos nos últimos 3 anos |
# ----------------------------------------------+

# Calculando a quantidade de produtos mais vendidos nos últimos 3 anos
print(dados[dados['Purchase Date'] > '02/06/2021'][dados['Quantity'] == max(dados['Quantity'])].shape[0])

# Calculando a quantidade de produtos vendidos por categoria nos últimos 3 anos
quant_3an = dados[dados['Purchase Date'] > '02/06/2021'][dados['Quantity'] == max(dados['Quantity'])][['Product Category', 'Quantity']].groupby('Product Category').sum().sort_values('Quantity', ascending=False)

# Construindo um gráfico de barras com a quantidade de produto por categoria
sns.barplot(data=quant_3an, x=pd.Categorical(quant_3an.index), y='Quantity', hue=pd.Categorical(quant_3an.index))
plt.title('Quantidade de produto vendido nos últimos 3 anos\npor categoria')
plt.xlabel('Categoria do Produto')
plt.ylabel('Quantidade Vendida')
plt.tight_layout()
plt.savefig('quant_prod_3an.jpg')

# Mostrando a tabela com os valores das categorias de produtos
quant_3an

# ---------------------------------------+
# (3) Produtos mais caros e mais baratos |
# ---------------------------------------+

# Calculando a quantidade de produtos com o preço máximo
print(dados[dados['Product Price'] == max(dados['Product Price'])].shape[0])

# Calculando a quantidade de produtos com o preço mínimo
print(dados[dados['Product Price'] == min(dados['Product Price'])].shape[0])

# Agrupando as compras com valores máximos em categorias
max_cat = dados[dados['Product Price'] == max(dados['Product Price'])][['Product Price', 'Product Category']].groupby('Product Category').mean()

# Agrupando as compras com valores mínimos em categorias
min_cat = dados[dados['Product Price'] == min(dados['Product Price'])][['Product Price', 'Product Category']].groupby('Product Category').mean()

# Concatenando os dois conjuntos de dados
precos_cat = pd.concat([max_cat, min_cat], axis=1)

# Ajustando o nome das colunas
precos_cat.columns = ['Máximo', 'Mínimo']

# Plotando gráfico com os máximos
sns.barplot(data=precos_cat, x=pd.Categorical(precos_cat.index), y='Máximo', hue=pd.Categorical(precos_cat.index))
plt.title('Valores máximos\npor categoria')
plt.xlabel('Categoria do Produto')
plt.ylabel('Preço Vendido')
plt.tight_layout()
plt.savefig('max_cat.jpg')

# Limpando a imagem do gráfico
plt.clf()

# Plotando gráfico com os mínimos
sns.barplot(data=precos_cat, x=pd.Categorical(precos_cat.index), y='Mínimo', hue=pd.Categorical(precos_cat.index))
plt.title('Valores mínimos\npor categoria')
plt.xlabel('Categoria do Produto')
plt.ylabel('Preço Vendido')
plt.ylim(0, 500)
plt.tight_layout()
plt.savefig('min_cat.jpg')

# Mostrando a tabela
precos_cat

# -----------------------------------------------------------+
# (4) Categorias mais e menos vendidas, e mais e menos caras |
# -----------------------------------------------------------+

# Agrupandos os preços médios por categoria de produto
med_preco = dados[['Product Category', 'Product Price']].groupby('Product Category').mean().apply(round)

# Plotando gráfico com os preços médios por categoria
sns.barplot(data=med_preco, x=pd.Categorical(med_preco.index), y='Product Price', hue=pd.Categorical(med_preco.index))
plt.title('Preço médio\npor categoria')
plt.xlabel('Categoria do Produto')
plt.ylabel('Preço Médio')
plt.ylim(0, 500)
plt.tight_layout()
plt.savefig('preco_med.jpg')

# Limptando a figura do gráfico
plt.clf()

# Agrupandos as quantidades por categoria de produto
sum_quant = dados[['Product Category', 'Quantity']].groupby('Product Category').sum().apply(round)

# Plotando gráfico com as quantidades médias por categoria
sns.barplot(data=sum_quant, x=pd.Categorical(sum_quant.index), y='Quantity', hue=pd.Categorical(sum_quant.index))
plt.title('Quantidade Total\npor categoria')
plt.xlabel('Categoria do Produto')
plt.ylabel('Quantidade Total Vendida')
plt.tight_layout()
plt.savefig('sum_quant.jpg')

# -----------------------------+
# (5) Produtos com o maior NPS |
# -----------------------------+

# Agrupandos os preços médios por categoria de produto
nps_med = dados[['Product Category', 'NPS']].groupby('Product Category').mean().apply(round)

# Plotando gráfico com os preços médios por categoria
sns.barplot(data=nps_med, x=pd.Categorical(nps_med.index), y='NPS', hue=pd.Categorical(nps_med.index))
plt.title('NPS médio\npor categoria')
plt.xlabel('Categoria do Produto')
plt.ylabel('NPS Médio')
plt.tight_layout()
plt.savefig('nps_med.jpg')

# ---------------------------------------------------------------------------+
# (6) Melhor público (gênero e idade) e canal para cada categoria de produto |
# ---------------------------------------------------------------------------+

# Calculando o ticket de cada compra
dados['Ticket'] = dados['Product Price'] * dados['Quantity']

# Calculando o ticket médio de cada cliente
tckt = dados[['Customer ID', 'Ticket']].groupby('Customer ID').mean()

# Salvando os tickets médios num dicionário
tckt_map = {c:v for c, v in zip(tckt.index, tckt.Ticket)} 

# Incluindo os tickets médios no conjunto de dados
dados['Tckt Médio'] = dados['Customer ID'].map(tckt_map)

# Retirando as colunas que não serão utilizadas
dados_mod = dados[['Product Category', 'Customer Age ', 'Gender', 'Source', 'Tckt Médio']]

# Removendo os valores duplicados
dados_mod.drop_duplicates(ignore_index=True, inplace=True)

# Transformando as variáveis categoricas em dummies
dados_mod_pronto = pd.get_dummies(dados_mod[['Customer Age ', 'Gender', 'Source', 'Tckt Médio']], drop_first=True, dtype=int)
dados_mod_pronto['Product Category'] = dados_mod['Product Category']

# Inserindo as interações no conjunto de dados
dados_mod_pronto['M_Idade'] = dados_mod_pronto['Gender_Male'] * dados_mod_pronto['Customer Age ']

dados_mod_pronto['Insta_Idade'] = dados_mod_pronto['Source_Instagram Campign'] * dados_mod_pronto['Customer Age ']
dados_mod_pronto['Org_Idade'] = dados_mod_pronto['Source_Organic Search'] * dados_mod_pronto['Customer Age ']
dados_mod_pronto['SEM_Idade'] = dados_mod_pronto['Source_SEM'] * dados_mod_pronto['Customer Age ']

dados_mod_pronto['Insta_M'] = dados_mod_pronto['Source_Instagram Campign'] * dados_mod_pronto['Gender_Male']
dados_mod_pronto['Org_M'] = dados_mod_pronto['Source_Organic Search'] * dados_mod_pronto['Gender_Male']
dados_mod_pronto['SEM_M'] = dados_mod_pronto['Source_SEM'] * dados_mod_pronto['Gender_Male']

dados_mod_pronto['Insta_M_Idade'] = dados_mod_pronto['Source_Instagram Campign'] * dados_mod_pronto['Gender_Male'] * dados_mod_pronto['Customer Age ']
dados_mod_pronto['Org_M_Idade'] = dados_mod_pronto['Source_Organic Search'] * dados_mod_pronto['Gender_Male'] * dados_mod_pronto['Customer Age ']
dados_mod_pronto['SEM_M_Idade'] = dados_mod_pronto['Source_SEM'] * dados_mod_pronto['Gender_Male'] * dados_mod_pronto['Customer Age ']

# Criando uma lista com as categorias dos produtos
cat = list(set(dados_mod['Product Category']))

# Investigando a relação entre o ticket médio e a idade do consumidor
# para cada categoria de produto
for i, c in enumerate(cat):

  plt.subplot(2, 2, i+1)

  sns.scatterplot(data=dados_mod[dados_mod['Product Category'] == c],
                x='Customer Age ', y='Tckt Médio', alpha=0.2)

  plt.title(f'{c}')
  plt.xlabel('Idade')
  plt.ylabel('Ticket Médio')

  plt.tight_layout()

plt.legend(loc='best', bbox_to_anchor=(1, 1))

plt.savefig('tckt_age.png')

plt.clf()

# Investigando a relação entre o ticket médio e o gênero do consumidor
# para cada categoria de produto
for i, c in enumerate(cat):

  plt.subplot(2, 2, i+1)

  sns.boxplot(data=dados_mod[dados_mod['Product Category'] == c],
                x='Gender', y='Tckt Médio')

  plt.title(f'{c}')
  plt.xlabel('Gênero')
  plt.ylabel('Ticket Médio')

  plt.tight_layout()

plt.savefig('tckt_gender.png')

plt.clf()

# Investigando a relação entre o ticket médio e o canal
# para cada categoria de produto
for i, c in enumerate(cat):

  plt.subplot(2, 2, i+1)

  sns.boxplot(data=dados_mod[dados_mod['Product Category'] == c],
                x='Source', y='Tckt Médio')

  plt.title(f'{c}')
  plt.xlabel('Canal')
  plt.xticks(rotation=25, fontsize=8)
  plt.ylabel('Ticket Médio')

  plt.tight_layout()

plt.savefig('tckt_canal.png')

plt.clf()

# Investigando a relação entre o ticket médio e a interação entre canal e gênero
# para cada categoria de produto
for i, c in enumerate(cat):

  plt.subplot(2, 2, i+1)

  if i == 3:

    sns.boxplot(data=dados_mod[dados_mod['Product Category'] == c],
                  x='Source', y='Tckt Médio', hue='Gender',
                    legend='brief')
    
  else:

    sns.boxplot(data=dados_mod[dados_mod['Product Category'] == c],
                  x='Source', y='Tckt Médio', hue='Gender',
                    legend=False)

  plt.title(f'{c}')
  plt.xlabel('Canal')
  plt.xticks(rotation=40, fontsize=8)
  plt.ylabel('Ticket Médio')

  plt.tight_layout()

plt.legend(loc='best', bbox_to_anchor=(1, 1), prop={'size':8})

plt.tight_layout()

plt.savefig('tckt_canalgenero.png')

plt.clf()

# Investigando a relação entre o ticket médio e a interação entre canal e idade
# para cada categoria de produto
for i, c in enumerate(cat):

  plt.subplot(2, 2, i+1)

  if i == 3:

    sns.scatterplot(data=dados_mod[dados_mod['Product Category'] == c],
                  x='Customer Age ', y='Tckt Médio', hue='Source',
                    alpha=0.2, legend='brief')
    
  else:

    sns.scatterplot(data=dados_mod[dados_mod['Product Category'] == c],
                  x='Customer Age ', y='Tckt Médio', hue='Source',
                    alpha=0.2, legend=False)

  plt.title(f'{c}')
  plt.xlabel('Idade')
  plt.xticks(fontsize=8)
  plt.ylabel('Ticket Médio')

  plt.tight_layout()

plt.legend(loc='best', bbox_to_anchor=(1, 1), prop={'size':8})

plt.tight_layout()

plt.savefig('tckt_canalidade.png')

plt.clf()

# Investigando a relação entre o ticket médio e a interação entre gênero e idade
# para cada categoria de produto
for i, c in enumerate(cat):

  plt.subplot(2, 2, i+1)

  if i == 3:

    sns.scatterplot(data=dados_mod[dados_mod['Product Category'] == c],
                  x='Customer Age ', y='Tckt Médio', hue='Gender',
                    alpha=0.2, legend='brief')
    
  else:

    sns.scatterplot(data=dados_mod[dados_mod['Product Category'] == c],
                  x='Customer Age ', y='Tckt Médio', hue='Gender',
                    alpha=0.2, legend=False)

  plt.title(f'{c}')
  plt.xlabel('Idade')
  plt.xticks(fontsize=8)
  plt.ylabel('Ticket Médio')

  plt.tight_layout()

plt.legend(loc='best', bbox_to_anchor=(1, 1), prop={'size':8})

plt.tight_layout()

plt.savefig('tckt_generoidade.png')

plt.clf()

# Investigando a relação entre o ticket médio e a interação entre canal, idade
# e gênero para cada categoria de produto
for i, c in enumerate(cat):

  plt.subplot(2, 2, i+1)

  if i == 3:

    sns.scatterplot(data=dados_mod[dados_mod['Product Category'] == c],
                  x='Customer Age ', y='Tckt Médio', hue='Source', size='Gender',
                    alpha=0.2, legend='brief')
    
  else:

    sns.scatterplot(data=dados_mod[dados_mod['Product Category'] == c],
                  x='Customer Age ', y='Tckt Médio', hue='Source', size='Gender',
                    alpha=0.2, legend=False)

  plt.title(f'{c}')
  plt.xlabel('Idade')
  plt.xticks(fontsize=8)
  plt.ylabel('Ticket Médio')

  plt.tight_layout()

plt.legend(loc='best', bbox_to_anchor=(1, 1), prop={'size':8})

plt.tight_layout()

plt.savefig('tckt_idadecanalgenero.png')

# Looping para iterar entre as categorias
for c in cat:

  # Separando a vairável resposta
  y = dados_mod_pronto[dados_mod_pronto['Product Category'] == c]['Tckt Médio']

  # Separando as covariáveis
  x = dados_mod_pronto[dados_mod_pronto['Product Category'] == c].drop(['Product Category', 'Tckt Médio'], axis=1)

  # Adicionando as constantes às covariáveis
  x = sm.add_constant(x)

  # Ajustando o modelo
  model = sm.OLS(y, x).fit()

  # Printando no console os resultados da regressão
  print(c)
  print(model.summary())
