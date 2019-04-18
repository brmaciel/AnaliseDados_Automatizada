# Processo de Avaliação de modelos utilizando Validação Cruzada

# Neste ponto deve-se ter definido os atributos previsores e a classe
# E já ter realizados todos os pre processamentos de dados necessários



########## ########## ########## ##########
# Funções Auxiliares
########## ########## ########## ##########
def create_test_model(classificador, index_train, index_test):
    # Cria e Testa o Modelo escolhido
    if classificador == 'naive bayes':
        modelo = GaussianNB()
    
    elif classificador == 'arvore':
        modelo = DecisionTreeClassifier()
    
    elif classificador == 'forest':
        modelo = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
    
    elif classificador == 'knn':
        modelo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    
    elif classificador == 'regressao':
        modelo = LogisticRegression(solver='liblinear')
    
    elif classificador == 'svm':
        modelo = SVC(kernel='rbf', random_state=1, C=2.0, gamma='auto')
    
    elif classificador == 'rna':
        modelo = MLPClassifier(max_iter=1000, tol=0.000001, solver='adam',
                           hidden_layer_sizes=(100), activation='relu',
                           batch_size=200, learning_rate_init=0.001)
    else:
        raise NameError ('Modelo escolhido nao esta na base de dados')
    
    modelo.fit(previsores[index_train], classe[index_train])
    previsoes = modelo.predict(previsores[index_test])

    return previsoes


def avalia_modelos(previsores, classe, classificador, n_seed, n_folds=10):
    print('\nClassificador: {}'.format(classificador.title()))
    
    resultado = []
    for seed in range(n_seed):
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        precisao = []
        for i_train, i_test in kfold.split(previsores, zeros(shape=(classe.shape[0], 1))):
            # Criacao e Teste do modelo
            previsoes = create_test_model(classificador, i_train, i_test)
            
            # Avaliacao do modelo
            score = accuracy_score(classe[i_test], previsoes)
            precisao.append(score)
        
        precisao = array(precisao)
        resultado.append(precisao.mean())
        print("seed {}: {}".format(seed, precisao.mean()))
    
    return resultado

def create_file(resultado, modelo):
    global result_dataframe
    
    vetor = []
    for i in range(len(resultado)):
        vetor.append([i, resultado[i]])
    vetor = DataFrame(vetor, columns=['seed', modelo])
    
    result_dataframe = result_dataframe.merge(vetor, on='seed')
    
    colunas = list(result_dataframe.columns)
    arquivo = DataFrame(result_dataframe, columns=colunas)
    arquivo.to_csv(nome_arquivo, index=False)

def executa_avaliacao(seeds, use_models):
    global model_result, model_name
    
    for modelo in use_models:
        model_name.append(all_models[modelo].title())
        result = avalia_modelos(previsores, classe, classificador=all_models[modelo], n_seed=seeds)
        model_result.append(result)
        average_result[all_models[modelo].title()] = Series(array(result).mean())
        
        create_file(result, modelo=all_models[modelo].title())
    
    average_result.sort_values(by=0, axis=1, ascending=False, inplace=True)

def view_accuracy():
    import matplotlib.pyplot as plt
    x = average_result.columns
    y = average_result.values.tolist()[0]
    plt.plot(x, y, 'o-')
    plt.title('Precisão')
    plt.xlabel('Modelos')
    plt.ylabel('Precisão')


    ##### ########## ########## ########## ########## ########## #####
    ##### ########## ########## ########## ########## ########## #####


# =====   Importação das Bibliotecas   ===== #
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from numpy import zeros, array
from pandas import DataFrame, Series



########## ########## ########## ##########
# Avaliação com Validação Cruzada
########## ########## ########## ##########

# =====   Carregamento dos atributos previsores e classe   ===== #
previsores = []
classe = []

    
# =====   Seleção dos modelos a utilizar   ===== #
# Selecionar os modelos alterando a variavel use_models
all_models = {1: 'naive bayes', 2: 'arvore', 3: 'forest',
              4: 'knn', 5: 'regressao', 6: 'svm', 7: 'rna'}
use_models = [1, 2, 5]
nome_arquivo = 'resultados.csv'


# =====   Coleta dos resultados e Criação do arquivo csv   ===== #
# Criacao do dataframe que armazenara os resultados
seeds = 30
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
