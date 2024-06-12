# imports==============================================
import pandas as pd
import numpy as np
from IPython.display import display, Image

from scipy import stats

from matplotlib import rcParams
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.vector_ar.var_model import VAR

# functions================================
# criação de features
def week_of_month(x):
    if x == 1:
        return 'primeira'
    elif x == 2:
        return 'segunda'
    elif x == 3:
        return 'terceira'
    elif x == 4:
        return 'quarta'
    else:
        return 'quinta'

def dia_da_semana(df, nome_coluna_data):
    nomes_dias = {
        'Monday': 'segunda',
        'Tuesday': 'terca',
        'Wednesday': 'quarta',
        'Thursday': 'quinta',
        'Friday': 'sexta',
        'Saturday': 'sabado',
        'Sunday': 'domingo'}

    df['dia_da_semana'] = df[nome_coluna_data].dt.day_name().map(nomes_dias)
    
    return df

def apply_rules(row):
    dia_da_semana = row['dia_da_semana']
    row['item_a'] = valores_item_a.get(dia_da_semana, 0)  
    row['item_b'] = valores_item_b.get(dia_da_semana, 0)
    row['item_c'] = valores_item_c.get(dia_da_semana, 0)
    row['item_d'] = valores_item_d.get(dia_da_semana, 0)
    return row

# eda 
def hipotese_1(df):
    # hipotese = Sexta vende-se 40% a mais em relação a segunda a quinta

    # Selecionar itens totais para sexta-feira
    sexta = df[df['dia_da_semana'] == 'sexta']['itens_totais']

    # Selecionar itens totais para segunda, terça, quarta e quinta
    outros_dias = df[df['dia_da_semana'].isin(['segunda', 'terca', 'quarta', 'quinta'])]['itens_totais']

    # Calcular a média de itens totais
    media_sexta = sexta.mean()
    media_outros_dias = outros_dias.mean()

    # Criar um DataFrame para plotagem
    data = {
        'Periodo': ['Sexta', 'Outros dias'],
        'Media de itens totais': [media_sexta, media_outros_dias]
    }

    df_plot = pd.DataFrame(data)

    # Criar o gráfico de barras
    fig = px.bar(df_plot, x='Periodo', y='Media de itens totais', color='Periodo',
                 labels={'Media de itens totais': 'Média de itens totais', 'Periodo': 'Período'},
                 title="Média de itens totais: Sexta vs Outros dias")

    # Calcular a diferença percentual
    diferenca_percentual = ((media_sexta - media_outros_dias) / media_outros_dias) * 100

    # Teste t de Student para comparar as médias
    u_statistic, p_value = stats.ttest_ind(sexta, outros_dias)

    alpha = 0.05
    resultado_relevancia = (
        "A diferença na média de itens totais entre sexta-feira e os outros dias da semana é estatisticamente relevante."
        if p_value < alpha else
        "A diferença na média de itens totais entre sexta-feira e os outros dias da semana não é estatisticamente relevante."
    )

    resultado = {
        'diferenca_percentual': diferenca_percentual,
        'valor_p': p_value,
        'resultado_relevancia': resultado_relevancia
    }

    return fig, resultado

def hipotese_2(df):
    # hipótese = Entre sexta a domingo, vende-se em média 60% a mais
    # Selecionar itens totais para sexta, sábado e domingo
    sexta_a_domingo = df[df['dia_da_semana'].isin(['sexta', 'sabado', 'domingo'])]['itens_totais']

    # Selecionar itens totais para segunda, terça, quarta e quinta
    outros_dias = df[df['dia_da_semana'].isin(['segunda', 'terca', 'quarta', 'quinta'])]['itens_totais']

    # Calcular a média de itens totais
    media_sexta_a_domingo = sexta_a_domingo.mean()
    media_outros_dias = outros_dias.mean()

    # Criar um DataFrame para plotagem
    data = {
        'Periodo': ['Sexta a Domingo', 'Outros dias'],
        'Media de itens totais': [media_sexta_a_domingo, media_outros_dias]
    }

    df_plot = pd.DataFrame(data)

    # Criar o gráfico de barras
    fig = px.bar(df_plot, x='Periodo', y='Media de itens totais', color='Periodo',
                 labels={'Media de itens totais': 'Média de itens totais', 'Periodo': 'Período'},
                 title="Média de itens totais: Sexta a Domingo vs Outros dias")

    # Calcular a diferença percentual
    diferenca_percentual = ((media_sexta_a_domingo - media_outros_dias) / media_outros_dias) * 100

    # Teste t de Student para comparar as médias
    u_statistic, p_value = stats.ttest_ind(sexta_a_domingo, outros_dias)

    alpha = 0.05
    resultado_relevancia = (
        "A diferença na média de itens totais entre sexta a domingo e os outros dias da semana é estatisticamente relevante."
        if p_value < alpha else
        "A diferença na média de itens totais entre sexta a domingo e os outros dias da semana não é estatisticamente relevante."
    )

    resultado = {
        'diferenca_percentual': diferenca_percentual,
        'valor_p': p_value,
        'resultado_relevancia': resultado_relevancia
    }

    return fig, resultado

def hipotese_3(df):
    # hipotese= Entre sábado a domingo, vende-se em média 100% a mais
    # Selecionar itens totais para sábado e domingo
    sabado_domingo = df[df['dia_da_semana'].isin(['sabado', 'domingo'])]['itens_totais']

    # Selecionar itens totais para os outros dias úteis
    outros_dias = df[df['dia_da_semana'].isin(['segunda', 'terca', 'quarta', 'quinta', 'sexta'])]['itens_totais']

    # Calcular a média de itens totais
    media_sabado_domingo = sabado_domingo.mean()
    media_outros_dias = outros_dias.mean()

    # Criar um DataFrame para plotagem
    data = {
        'Periodo': ['Sábado e Domingo', 'Outros dias'],
        'Media de itens totais': [media_sabado_domingo, media_outros_dias]
    }

    df_plot = pd.DataFrame(data)

    # Criar o gráfico de barras
    fig = px.bar(df_plot, x='Periodo', y='Media de itens totais', color='Periodo',
                 labels={'Media de itens totais': 'Média de itens totais', 'Periodo': 'Período'},
                 title="Média de itens totais: Sábado e Domingo vs Outros dias")

    # Calcular a diferença percentual
    diferenca_percentual = ((media_sabado_domingo - media_outros_dias) / media_outros_dias) * 100

    # Teste t de Student para comparar as médias
    u_statistic, p_value = stats.ttest_ind(sabado_domingo, outros_dias)

    alpha = 0.05
    resultado_relevancia = (
        "A diferença na média de itens totais entre sábado e domingo e os outros dias da semana é estatisticamente relevante."
        if p_value < alpha else
        "A diferença na média de itens totais entre sábado e domingo e os outros dias da semana não é estatisticamente relevante."
    )

    resultado = {
        'diferenca_percentual': diferenca_percentual,
        'valor_p': p_value,
        'resultado_relevancia': resultado_relevancia
    }

    return fig, resultado

def hipotese_4(df):
    # hipótese = Às quintas, o item b vende 30% menos do que os outros itens
    # Média de vendas do Item B nas quintas-feiras
    media_quinta_item_b = df[df['dia_da_semana'] == 'quinta']['item_b'].mean()

    # Média de vendas dos outros itens nas quintas-feiras
    media_quinta_outros_itens = df[df['dia_da_semana'] == 'quinta'][['item_a', 'item_c', 'item_d']].mean().mean()

    # Criação do DataFrame para o gráfico de barras
    data = {
        'Item': ['Item B', 'Outros Itens'],
        'Média de Vendas': [media_quinta_item_b, media_quinta_outros_itens]
    }
    df_bar = pd.DataFrame(data)

    # Plot do gráfico de barras
    fig_bar = px.bar(df_bar, x='Item', y='Média de Vendas', 
                     labels={'Média de Vendas': 'Média de Vendas', 'Item': 'Item'},
                     title='Média de Vendas do Item B vs Outros Itens nas Quintas-feiras')

    # Cálculo da diferença percentual
    diferenca_percentual = ((media_quinta_outros_itens - media_quinta_item_b) / media_quinta_outros_itens) * 100

    # Realizando o teste t
    quinta_item_b = df[df['dia_da_semana'] == 'quinta']['item_b']
    quinta_outros_itens = df[df['dia_da_semana'] == 'quinta'][['item_a', 'item_c', 'item_d']].values.flatten()

    t_statistic, p_value = stats.ttest_ind(quinta_item_b, quinta_outros_itens)

    alpha = 0.05
    resultado_relevancia = (
        "A diferença na média de vendas do Item B nas quintas-feiras em comparação com os outros itens nas quintas-feiras é estatisticamente relevante."
        if p_value < alpha else
        "A diferença na média de vendas do Item B nas quintas-feiras em comparação com os outros itens nas quintas-feiras não é estatisticamente relevante."
    )

    resultado = {
        'diferenca_percentual': diferenca_percentual,
        'valor_p': p_value,
        'resultado_relevancia': resultado_relevancia
    }

    return fig_bar, resultado

def hipotese_5(df):
    # hipotese=Às sextas, os item a e c vendem 50% menos do que os itens b e d
    # Média de vendas dos itens A e C às sextas-feiras
    media_sexta_itens_ac = df[df['dia_da_semana'] == 'sexta'][['item_a', 'item_c']].mean().mean()

    # Média de vendas dos itens B e D às sextas-feiras
    media_sexta_itens_bd = df[df['dia_da_semana'] == 'sexta'][['item_b', 'item_d']].mean().mean()

    # Criação do DataFrame para o gráfico de barras
    data = {
        'Itens': ['Itens A e C', 'Itens B e D'],
        'Média de Vendas às Sextas': [media_sexta_itens_ac, media_sexta_itens_bd]
    }
    df_bar = pd.DataFrame(data)

    # Plot do gráfico de barras
    fig_bar = px.bar(df_bar, x='Itens', y='Média de Vendas às Sextas', 
                     labels={'Média de Vendas às Sextas': 'Média de Vendas', 'Itens': 'Itens'},
                     title='Média de Vendas às Sextas: Itens A e C vs Itens B e D')

    # Cálculo da diferença percentual
    diferenca_percentual = ((media_sexta_itens_bd - media_sexta_itens_ac) / media_sexta_itens_bd) * 100

    # Realizando o teste t
    sexta_itens_ac = df[df['dia_da_semana'] == 'sexta'][['item_a', 'item_c']].values.flatten()
    sexta_itens_bd = df[df['dia_da_semana'] == 'sexta'][['item_b', 'item_d']].values.flatten()

    t_statistic, p_value = stats.ttest_ind(sexta_itens_ac, sexta_itens_bd)

    alpha = 0.05
    resultado_relevancia = (
        "A diferença na média de vendas dos Itens A e C às sextas-feiras em comparação com os Itens B e D é estatisticamente relevante."
        if p_value < alpha else
        "A diferença na média de vendas dos Itens A e C às sextas-feiras em comparação com os Itens B e D não é estatisticamente relevante."
    )

    resultado = {
        'diferenca_percentual': diferenca_percentual,
        'valor_p': p_value,
        'resultado_relevancia': resultado_relevancia
    }

    return fig_bar, resultado

# load data==============================================
item_pedido=pd.read_excel('../data/ITEM_PEDIDO-_2_ _.xlsx',index_col=0)
itens=pd.read_excel('../data/ITENS-_3___.xlsx',index_col=0)
pedido=pd.read_excel('../data/PEDIDO-_1__.xlsx',index_col=0)

# união dos dataframes===========================================================
# criando uma tabela auxiliar a partir da tabela 'item_pedido' com a coluna ID ITEM quebrada em quatro colunas para cada tipo de item
pivot_table = item_pedido.pivot_table(index='ID_PEDIDO', columns='ID_ITEM', values='QUANTIDADE', fill_value=0).reset_index()

# unindo a tabela 'pedido' com a tabela auxiliar através da coluna 'ID_PEDIDO'
pedido = pedido.merge(pivot_table[['ID_PEDIDO','item A', 'item B', 'item C', 'item D']], on='ID_PEDIDO', how='left')

# agrupando a tabela pedido pela coluna DATA a fim de ter a soma total de cada item por dia 
pedido_grouped = pedido.groupby('DATA').agg({'ID_PEDIDO': 'first',  # mantém o primeiro ID_PEDIDO
                                            'item A': 'sum',
                                            'item B': 'sum',
                                            'item C': 'sum',
                                            'item D': 'sum'}).reset_index()

# dropando coluna que não será utilizada 
pedido_grouped.drop(columns=['ID_PEDIDO'],inplace=True)
df1=pedido_grouped

# renomear e mudar tipos==========================================================
df1.rename(columns={'DATA':'data','item A':'item_a','item B':'item_b','item C':'item_c','item D':'item_d'},inplace=True)
df1[['item_a', 'item_b', 'item_c', 'item_d']]=df1[['item_a', 'item_b', 'item_c', 'item_d']].astype(int)

# Separação entre treino, teste e validação=======================================
X = df1.drop(labels=['item_a','item_b','item_c','item_d'], axis=1)
y = df1[['item_a','item_b','item_c','item_d']]

# TimeSeriesSplit em treino, validação e teste
tscv = TimeSeriesSplit(n_splits=3)

# iterar sobre os splits
for train_index, test_index in tscv.split(X):
    
    # dividir em treino e teste
    X_train_full, X_test_full = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # dividir o treino em treino e validação (70/30)
    split_idx = int(len(X_train_full) * 0.7)
    X_train_full, X_val_full = X_train_full[:split_idx], X_train_full[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]

X_val_full.reset_index(drop=True,inplace=True)
X_test_full.reset_index(drop=True,inplace=True)
y_val.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)

df1 = df1[df1['data'] <= '2021-08-12']

# feature engineering================================================================
df2=df1.copy()

# soma total de cada item
df2['itens_totais'] = df2[['item_a', 'item_b', 'item_c', 'item_d']].sum(axis=1)

# dia da semana
df2 = dia_da_semana(df2.loc[:, :], 'data')
X_train_full = dia_da_semana(X_train_full.loc[:, :], 'data')
X_val_full = dia_da_semana(X_val_full.loc[:, :], 'data')
X_test_full = dia_da_semana(X_test_full.loc[:, :], 'data')

# dia do mês
df2.loc[:, 'dia'] = df2['data'].dt.day

# tipo do dia
df2.loc[:, 'tipo_dia'] = df2['data'].dt.weekday.apply(lambda x: 'util' if x < 5 else 'fds')
X_train_full.loc[:, 'tipo_dia'] = X_train_full['data'].dt.weekday.apply(lambda x: 'util' if x < 5 else 'fds')
X_val_full.loc[:, 'tipo_dia'] = X_val_full['data'].dt.weekday.apply(lambda x: 'util' if x < 5 else 'fds')
X_test_full.loc[:, 'tipo_dia'] = X_test_full['data'].dt.weekday.apply(lambda x: 'util' if x < 5 else 'fds')

# semana do mês
df2.loc[:, 'semana_do_mes'] = df2['data'].dt.day.apply(lambda x: week_of_month((x - 1) // 7 + 1))
X_train_full.loc[:, 'semana_do_mes'] = X_train_full['data'].dt.day.apply(lambda x: week_of_month((x - 1) // 7 + 1))
X_val_full.loc[:, 'semana_do_mes'] = X_val_full['data'].dt.day.apply(lambda x: week_of_month((x - 1) // 7 + 1))
X_test_full.loc[:, 'semana_do_mes'] = X_test_full['data'].dt.day.apply(lambda x: week_of_month((x - 1) // 7 + 1))

# médias móveis
df2['itens_totais_mm'] = df2['itens_totais'].rolling(window=7).mean()
df2['item_a_mm'] = df2['item_a'].rolling(window=7).mean()
df2['item_b_mm'] = df2['item_b'].rolling(window=7).mean()
df2['item_c_mm'] = df2['item_c'].rolling(window=7).mean()
df2['item_d_mm'] = df2['item_d'].rolling(window=7).mean()

# eda===============================================
df4=df2.copy()

# (somente as hipóteses mais importantes do notebook)

# Sexta vende-se 40% a mais em relação a segunda a quinta
fig1, resultado1 = hipotese_1(df4)

# Entre sexta a domingo, vende-se em média 60% a mais
fig2, resultado2 = hipotese_2(df4)

# Entre sábado a domingo, vende-se em média 100% a mais
fig3, resultado3 = hipotese_3(df4)

# Às quintas, o item b vende 30% menos do que os outros itens
fig4, resultado4 = hipotese_4(df4)

# Às sextas, os item a e c vendem 50% menos do que os itens b e d
fig5, resultado5 = hipotese_5(df4)

# preparação dos dados========================================
df5=df4.copy()

# encoders
cols = ['dia_da_semana', 'tipo_dia', 'semana_do_mes']

enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

train_enc = enc.fit_transform(X_train_full[cols])
val_enc = enc.transform(X_val_full[cols])
test_enc = enc.transform(X_test_full[cols])

names = enc.get_feature_names_out(cols)

X_train= pd.DataFrame(train_enc, columns=names)
X_val= pd.DataFrame(val_enc, columns=names)
X_test = pd.DataFrame(test_enc, columns=names)

# seleção de features=================================================
df6=df5.copy() 

# dataset normal
final_cols=['dia_da_semana_segunda','dia_da_semana_sexta','dia_da_semana_sabado','dia_da_semana_domingo']
X_train=X_train[final_cols]
X_val=X_val[final_cols]
X_test=X_test[final_cols]

# dataset para algoritmos de time series
X_train_ts=X_train[final_cols]
X_val_ts=X_val[final_cols]
X_test_ts=X_test[final_cols]

X_train_ts[['item_a','item_b','item_c','item_d']]=y_train[['item_a','item_b','item_c','item_d']]
X_val_ts[['item_a','item_b','item_c','item_d']]=y_val[['item_a','item_b','item_c','item_d']]
X_test_ts[['item_a','item_b','item_c','item_d']]=y_test[['item_a','item_b','item_c','item_d']]

X_train_ts.loc[:,'data'] = X_train_full['data']
X_val_ts.loc[:,'data'] = X_val_full['data']
X_test_ts.loc[:,'data'] = X_test_full['data']

X_train_ts.set_index('data', inplace=True)
X_val_ts.set_index('data', inplace=True)
X_test_ts.set_index('data', inplace=True)

X_train_ts.index = pd.DatetimeIndex(X_train_ts.index).to_period('D')
X_val_ts.index = pd.DatetimeIndex(X_val_ts.index).to_period('D')
X_test_ts.index = pd.DatetimeIndex(X_test_ts.index).to_period('D')

X_train_ts=pd.concat([X_train_ts,X_val_ts])

# machine learning============================================================
df7=df6.copy()

# regressão linear 
reg = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)

reg_pred = reg.predict(X_test)

reg_mae = mean_absolute_error(y_test, reg_pred)
a_reg_mae = mean_absolute_error(y_test['item_a'], reg_pred[:, 0])
b_reg_mae = mean_absolute_error(y_test['item_b'], reg_pred[:, 1])
c_reg_mae = mean_absolute_error(y_test['item_c'], reg_pred[:, 2])
d_reg_mae = mean_absolute_error(y_test['item_d'], reg_pred[:, 3])

# print("Regression - MAE geral: ", reg_mae)
# print("Regression - MAE Item A: ", a_reg_mae)
# print("Regression - MAE Item B: ", b_reg_mae)
# print("Regression - MAE Item C: ", c_reg_mae)
# print("Regression - MAE Item D: ", d_reg_mae)

# decision tree e random forest 
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)
forest_pred = forest_model.predict(X_test)
 
tree_mae = mean_absolute_error(y_test, tree_pred)
a_tree_mae = mean_absolute_error(y_test['item_a'], tree_pred[:, 0])
b_tree_mae = mean_absolute_error(y_test['item_b'], tree_pred[:, 1])
c_tree_mae = mean_absolute_error(y_test['item_c'], tree_pred[:, 2])
d_tree_mae = mean_absolute_error(y_test['item_d'], tree_pred[:, 3])

forest_mae = mean_absolute_error(y_test, forest_pred)
a_forest_mae = mean_absolute_error(y_test['item_a'], forest_pred[:, 0])
b_forest_mae = mean_absolute_error(y_test['item_b'], forest_pred[:, 1])
c_forest_mae = mean_absolute_error(y_test['item_c'], forest_pred[:, 2])
d_forest_mae = mean_absolute_error(y_test['item_d'], forest_pred[:, 3])

# print("Decision Tree - MAE geral: ", tree_mae)
# print("Decision Tree - MAE Item a: ", a_tree_mae)
# print("Decision Tree - MAE Item b: ", b_tree_mae)
# print("Decision Tree - MAE Item c: ", c_tree_mae)
# print("Decision Tree - MAE Item d: ", d_tree_mae)

# print('\n'"Random Forest Model - MAE geral: ", forest_mae)
# print("Random Forest - MAE Item a: ", a_forest_mae)
# print("Random Forest - MAE Item b: ", b_forest_mae)
# print("Random Forest - MAE Item c: ", c_forest_mae)
# print("Random Forest - MAE Item d: ", d_forest_mae)

# time series
model = VAR(X_train_ts)
model_fit = model.fit()

pred = model_fit.forecast(X_train_ts.values, steps=len(X_test_ts))

pred_df = pd.DataFrame(pred, index=X_test_ts.index, columns=X_train_ts.columns)

mae_a = mean_absolute_error(X_test_ts['item_a'], pred_df['item_a'])
mae_b = mean_absolute_error(X_test_ts['item_b'], pred_df['item_b'])
mae_c = mean_absolute_error(X_test_ts['item_c'], pred_df['item_c'])
mae_d = mean_absolute_error(X_test_ts['item_d'], pred_df['item_d'])
mae_geral = (mae_a + mae_b + mae_c + mae_d) / 4

# print(f'Time series - MAE Geral: {mae_geral}')
# print(f'Time series - MAE Item a: {mae_a}')
# print(f'Time series - MAE Item b: {mae_b}')
# print(f'Time series - MAE Item c: {mae_c}')
# print(f'Time series - MAE Item d: {mae_d}')


# modelo baseline (MODELO ESCOLHIDO) (mediana dos itens de acordo com o dia da semana)
median_by_day = pd.pivot_table(df7, values=['item_a', 'item_b', 'item_c', 'item_d'], index='dia_da_semana', aggfunc='median')
median_by_day.reset_index(inplace=True)

y_pred=y_test.copy()
y_pred['dia_da_semana']=X_test_full['dia_da_semana']

valores_item_a = median_by_day.set_index('dia_da_semana')['item_a'].to_dict()
valores_item_b = median_by_day.set_index('dia_da_semana')['item_b'].to_dict()
valores_item_c = median_by_day.set_index('dia_da_semana')['item_c'].to_dict()
valores_item_d = median_by_day.set_index('dia_da_semana')['item_d'].to_dict()

y_pred = y_pred.apply(apply_rules, axis=1)

mae_item_a = mean_absolute_error(y_test['item_a'], y_pred['item_a'])
mae_item_b = mean_absolute_error(y_test['item_b'], y_pred['item_b'])
mae_item_c = mean_absolute_error(y_test['item_c'], y_pred['item_c'])
mae_item_d = mean_absolute_error(y_test['item_d'], y_pred['item_d'])

mae_geral = (mae_item_a + mae_item_b + mae_item_c + mae_item_d) / 4

# print("MAE geral:", mae_geral)
# print("MAE Item a:", mae_item_a)
# print("MAE Item b:", mae_item_b)
# print("MAE Item c:", mae_item_c)
# print("MAE Item d:", mae_item_d)