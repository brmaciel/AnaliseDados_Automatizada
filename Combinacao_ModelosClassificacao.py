# Processo de Combinação de modelos

# Neste ponto deve-se ter definido os modelos com as melhores precisoes



########## ########## ########## ##########
# Funções Auxiliares
########## ########## ########## ##########
def create_models():
    # Cria os modelos a serem utilizados e os retorna
    rna_model = MLPClassifier(max_iter=1000, tol=0.000001, solver='adam',
                           hidden_layer_sizes=(100), activation='relu',
                           batch_size=200, learning_rate_init=0.001)
    rna_model.fit(previsores, classe)

    svm_model = SVC(kernel='rbf', random_state=1, C=2.0, gamma='auto')
    svm_model.fit(previsores, classe)

    knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn_model.fit(previsores, classe)
    
    return rna_model, svm_model, knn_model

def save_models(modelos, nomes):
    # Salva os modelos em disco
    for modelo, nome in zip(modelos, nomes):
        arquivo = nome + '.sav'
        pickle.dump(modelo, open(arquivo, 'wb'))

def analiza_registro(registro, nome_modelo):
    # Realiza as previsoes para cada um dos modelos
    # retorna as previsoes em um DataFrame
    dados = np.zeros((len(registro), len(nome_modelo)))
    previsoes = DataFrame(dados, columns=nome_modelo)

    for modelo, nome in zip(classificadores, nome_modelo):
        print(modelo.predict(registro))
        previsoes[nome] = modelo.predict(registro)
    
    return previsoes

def combine_models(previsoes):
    # Define a classe do registro combinando as previsoes
    # utiliza o metodo de pluralidade (mais votos)
    # retorna as classificacoes definidas pelo metodo e
    #         os registros concatenanto as classificacoes com os atributos
    classificacoes = np.zeros(len(previsoes))
    classificacoes = classificacoes.reshape(-1,1)

    for i in range(len(previsoes)):
        lista = previsoes.iloc[i].values
        elemento, n_repeat = np.unique(lista, return_counts=True)
        aux = dict(zip(n_repeat, elemento))
        classificacoes[i] = aux[max(aux.keys())]
    
    registros = np.concatenate((novo_registro, classificacoes), axis=1)
    
    return registros, classificacoes
    
    ##### ########## ########## ########## ########## ########## #####
    ##### ########## ########## ########## ########## ########## #####


# =====   Importação das Bibliotecas   ===== #
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import pickle
from pandas import DataFrame, Series
import numpy as np



########## ########## ########## ##########
# Combinação dos Modelos
########## ########## ########## ##########

# =====   Carregamento dos atributos previsores e classe   ===== #
previsores = []
classe = []
list_atributos = list(base.columns) # coleta nome dos atributos da base de dados


# =====   Criação dos modelos a utilizar   ===== #
list_models = ['modelo_rna', 'modelo_svm', 'modelo_knn']

# OPÇÃO 1:
# Cria os modelos de classificacao
classificadores = create_models(proba=True)
# Salva os modelos em disco
save_models(classificadores, nomes=list_models)

# OPÇÃO 2:
# Carrega os modelo salvos em disco
model_rna = pickle.load(open('modelo_rna.sav', 'rb'))
model_svm = pickle.load(open('modelo_svm.sav', 'rb'))
model_knn = pickle.load(open('modelo_knn.sav', 'rb'))
classificadores = [model_rna, model_svm, model_knn]

# Avaliação dos modelos carregados
precisoes = DataFrame(columns=list_models)
for i in range(len(classificadores)):
    precisoes[list_models[i]] = Series(classificadores[i].score(previsores, classe))


# =====   Teste de Classificação de novo registro   ===== #
novo_registro = [[3, 3, 2, 2], [5.4, 3, 4.5, 1.5],
                 [5.9, 3.2, 4.8, 1.8], [6.3, 2.5, 4.9, 1.5],
                 [4.9, 2.5, 4.5, 1.7], [6, 2.2, 5, 1.5], [6.3, 2.8, 5.1, 1.5]]
novo_registro = np.array(novo_registro)

novo_registro = previsores
previsoes = analiza_registro(novo_registro, list_models)


# =====   Combinação dos Classificadores   ===== #
registros, _ = combine_models(previsoes)
registros = DataFrame(registros, columns=list_atributos)
