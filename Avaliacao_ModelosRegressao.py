# Processo de Avaliação de modelos utilizando Validação Cruzada

# Neste ponto deve-se ter definido os atributos previsores e a classe
# E já ter realizados todos os pre processamentos de dados necessários



########## ########## ########## ##########
# Funções Auxiliares
########## ########## ########## ##########
def create_test_model(regressor, index_train, index_test):
    # Cria e Testa o Modelo escolhido
    if regressor == 'regressao':
        modelo = LinearRegression()
    
    elif regressor == 'arvore':
        modelo = DecisionTreeRegressor(random_state=1)
    
    elif regressor == 'forest':
        modelo = RandomForestRegressor(n_estimators=40, random_state=0)
    
    elif regressor == 'svm':
        modelo = SVR(kernel='rbf', C=2.0, gamma='auto')
    
    elif regressor == 'rna':
        modelo = MLPRegressor(max_iter=1000, tol=0.000001, solver='adam',
                           hidden_layer_sizes=(9,9))
    else:
        raise NameError ('Modelo escolhido nao esta na base de dados')
    
    modelo.fit(previsores[index_train], classe[index_train])
    previsoes = modelo.predict(previsores[index_test])
    previsoes = scaler.inverse_transform(previsoes)
    
    score = modelo.score(previsores[index_test], classe[index_test])

    return previsoes, score

def avalia_modelos(previsores, classe, regressor, n_seed, n_folds=10):
    print('\nRegressor: {}'.format(regressor.title()))
    
    resultado = []
    for seed in range(n_seed):
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        precisao = []
        for i_train, i_test in kfold.split(previsores, zeros(shape=(classe.shape[0], 1))):
            # Criacao e Teste do modelo
            previsoes, score = create_test_model(regressor, i_train, i_test)
            
            
            # Avaliacao do modelo
            erro_medio = -mean_absolute_error(scaler.inverse_transform(classe[i_test]), previsoes)
            precisao.append([score, erro_medio]) # score = correlacao
        
        precisao = array(precisao)
        resultado.append(precisao.mean(axis=0))
        print("seed {}: {:.5f} - {:.5f}".format(seed, precisao.mean(axis=0)[0], -precisao.mean(axis=0)[1]))
    
    return resultado

def create_file(resultado, modelo):
    global result_dataframe, vetor
    colunas = ['seed', '{} score'.format(modelo), '{} erro'.format(modelo)]
    
    vetor = []
    for i in range(len(resultado)):
        vetor.append([i, resultado[i][0], resultado[i][1]])
    vetor = DataFrame(vetor, columns=colunas)
    
    result_dataframe = result_dataframe.merge(vetor, on='seed')
    
    colunas = list(result_dataframe.columns)
    arquivo = DataFrame(result_dataframe, columns=colunas)
    arquivo.to_csv(nome_arquivo, index=False)

def executa_avaliacao(seeds, use_models):
    global model_result, model_name
    
    for modelo in use_models:
        model_name.append(all_models[modelo].title())
        result = avalia_modelos(previsores, classe, regressor=all_models[modelo], n_seed=seeds)
        model_result.append(result)
        average_result[all_models[modelo].title()] = Series(array(result).mean(axis=0))
        
        create_file(result, modelo=all_models[modelo].title())
    
    average_result.sort_values(by=0, axis=1, ascending=False, inplace=True)
    average_result.index = ['score', 'erro_medio']

def view_accuracy():
    fig, ax1 = plt.subplots(figsize=(9,6))
    ax1.plot(average_result.columns, average_result.loc['score'],
             color='b', marker='o')
    ax1.set_ylabel('Score', color='b')
    for lb in ax1.get_yticklabels():
        lb.set_color('b')
    plt.ylim(top=max(average_result.loc['score'])+0.025)
    plt.grid(axis='y', linewidth="0.5")
    
    ax2 = ax1.twinx()
    ax2.plot(average_result.columns, average_result.loc['erro_medio'],
             color='red', marker='o')
    plt.yticks(list(range(min(average_result.loc['erro_medio']),
                          max(average_result.loc['erro_medio']), 5000)))
    ax2.set_ylabel('Erro Medio', color='red')
    for lb in ax2.get_yticklabels():
        lb.set_color('red')

    plt.title("Score e Erro Medio")

    ##### ########## ########## ########## ########## ########## #####
    ##### ########## ########## ########## ########## ########## #####


# =====   Importação das Bibliotecas   ===== #
from pandas import read_csv, DataFrame, Series
from numpy import zeros, array
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

# Desabilita os warnings
import warnings
warnings.filterwarnings('ignore')



########## ########## ########## ##########
# Avaliação com Validação Cruzada
########## ########## ########## ##########

# =====   Carregamento dos atributos previsores e classe   ===== #
previsores = []
classe = []


# =====   Seleção dos modelos a utilizar   ===== #
all_models = {1: 'regressao', 2: 'arvore', 3: 'forest',
              4: 'svm', 5: 'rna'}
use_models = [1, 2, 3, 4, 5]
nome_arquivo = 'resultados.csv' # nome do arquivo a serem salvos os resultados


# =====   Coleta dos resultados e Criação do arquivo csv   ===== #
seeds = 10
result_dataframe = DataFrame(list(range(seeds)), columns=['seed'])
average_result = DataFrame()

model_result = []
model_name = []

# Coletar os resultados associando-os aos respectivos modelos
# e Armazena-los em disco em formato csv
executa_avaliacao(seeds, use_models)
#executa_avaliacao(seeds, [3]) # Testar 1 unico modelo

# Visualização Grafica das Precisoes
view_accuracy()