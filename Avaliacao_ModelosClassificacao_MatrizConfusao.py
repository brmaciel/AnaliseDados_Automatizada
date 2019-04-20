# Processo de Avaliação de modelos utilizando Validação Cruzada

# Neste ponto deve-se ter definido os atributos previsores e a classe
# E já ter realizados todos os pre processamentos de dados necessários


# ATENÇÃO: Funciona apenas para base de dados com 2 valores de classe


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
    result_matriz = []
    for seed in range(n_seed):
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        precisao = []
        matrizes = []
        for i_train, i_test in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):
            # Criacao e Teste do modelo
            previsoes = create_test_model(classificador, i_train, i_test)
            
            # Avaliacao do modelo
            score = accuracy_score(classe[i_test], previsoes)
            matriz = confusion_matrix(classe[i_test], previsoes)
            
            precisao.append(score)
            matrizes.append(matriz)
        
        precisao = np.array(precisao)
        resultado.append(precisao.mean())
        result_matriz.append(np.mean(matrizes, axis=0))
        
        print("seed {}: {}".format(seed, precisao.mean()))
    
    return resultado, result_matriz

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
    global model_result, model_name, model_matrix_result
    
    for modelo in use_models:
        model_name.append(all_models[modelo].title())
        result, result_matriz = avalia_modelos(previsores, classe, classificador=all_models[modelo], n_seed=seeds)
        model_result.append(result)
        model_matrix_result.append(np.mean(result_matriz, axis=0))
        average_result[all_models[modelo].title()] = Series(np.array(result).mean())
        
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

def view_confusion_matrix(classe0='Classe 0', classe1='Classe 1', detail_models=False):
    # Divisão da matriz de confusão em listas
    true_class0 = []
    false_class1 = []
    false_class0 = []
    true_class1 = []
    for matriz in model_matrix_result:
        true_class0.append(matriz[0][0])
        false_class1.append(matriz[0][1])
        false_class0.append(matriz[1][0])
        true_class1.append(matriz[1][1])
    
    # Manipulações necessarias para plotar barras lado a lado
    bar_width = 0.4
    posicao_bar = list(np.arange(bar_width, len(model_name)+bar_width))
    posicao_ticks = list(np.arange(bar_width/2, len(model_name)+bar_width/2))
    
    bar_width2 = 0.2
    posicao_bar1 = list(np.arange(bar_width2, len(detail_models)+bar_width2))
    posicao_bar2 = list(np.arange(bar_width2*2, len(detail_models)+bar_width2*2))
    posicao_bar3 = list(np.arange(bar_width2*3, len(detail_models)+bar_width2*3))
    posicao_ticks2 = list(np.arange(bar_width2*1.5, len(detail_models)+bar_width2*1.5))
    
    ##### === Plot dos Gráficos === #####
    plt.figure(figsize=(12,11))
    
    # Subplot da Classe 0
    plt.subplot(2, 1, 1)
    plt.bar(model_name, true_class0, width=bar_width, label='Verdadeiro {}'.format(classe0))
    plt.bar(posicao_bar, false_class0, width=bar_width, label='Falso {}'.format(classe0), color='grey')
    plt.xticks(ticks=posicao_ticks)
    plt.legend()
    plt.grid(axis='y', linewidth="0.5")
    
    j = 0
    for i in list(np.arange(0, 7)):
        try:
            texto = '{:.1f}'.format(true_class0[j])
            plt.text(i-0.1, true_class0[j]-3.5, texto, verticalalignment='bottom')
            
            texto = '{:.1f}'.format(false_class0[j])
            plt.text(i+0.3, false_class0[j]-3.5, texto, verticalalignment='bottom', color='white')
        except:
            pass
        finally:
            j += 1
    
    # Subplot da Classe 1
    plt.subplot(2, 1, 2)
    plt.bar(model_name, true_class1, width=bar_width, label='Verdadeiro {}'.format(classe1), color='orange')
    plt.bar(posicao_bar, false_class1, width=bar_width, label='Falso {}'.format(classe1), color='grey')
    plt.xticks(ticks=posicao_ticks)
    plt.legend()
    plt.grid(axis='y', linewidth="0.5")
    
    j = 0
    for i in list(np.arange(0, 7)):
        try:
            texto = '{:.1f}'.format(true_class1[j])
            plt.text(i-0.1, true_class1[j]-1, texto, verticalalignment='bottom')
            
            texto = '{:.1f}'.format(false_class1[j])
            plt.text(i+0.3, false_class1[j]-1, texto, verticalalignment='bottom', color='white')
        except:
            pass
        finally:
            j += 1
    
    plt.tight_layout()
    
    
    # Plota grafico abordando somente os modelos selecionados
    if detail_models != False:
        # Seleciona os dados apenas dos modelos interessados
        x = []
        y_true_class0 = []
        y_false_class0 = []
        y_true_class1 = []
        y_false_class1 = []
        for i in detail_models:
            if i in use_models:
                indice = use_models.index(i)
                x.append(model_name[indice])
                y_true_class0.append(true_class0[indice])
                y_false_class0.append(false_class0[indice])
                y_true_class1.append(true_class1[indice])
                y_false_class1.append(false_class1[indice])
        
        
        plt.figure(figsize=(12, 9))
        plt.bar(x, y_true_class0, width=bar_width2, label='Verdadeiro {}'.format(classe0))
        plt.bar(posicao_bar1, y_false_class0, width=bar_width2, label='Falso {}'.format(classe0), color='grey')
        plt.bar(posicao_bar2, y_true_class1, width=bar_width2, label='Verdadeiro {}'.format(classe1), color='orange')
        plt.bar(posicao_bar3, y_false_class1, width=bar_width2, label='Falso {}'.format(classe1), color='#cccccc')
        plt.xticks(ticks=posicao_ticks2)
        plt.ylabel('Quantidade de Registros')
        plt.legend()
        plt.grid(axis='y', linewidth='0.5')
        
        # Adiciona valores (%) em texto às barras
        j = 0
        for i in list(np.arange(0, 7)):
            try:
                texto = '{:.2f}%'.format(y_true_class0[j] / (y_true_class0[j] + y_false_class0[j]) * 100)
                plt.text(i-0.07, y_true_class0[j]-3, texto, verticalalignment='bottom', fontsize='15')
                
                texto = '{:.2f}%'.format(y_false_class0[j] / (y_true_class0[j] + y_false_class0[j]) * 100)
                plt.text(i+0.12, y_false_class0[j]-3, texto, verticalalignment='bottom', fontsize='15')
                
                texto = '{:.2f}%'.format(y_true_class1[j] / (y_true_class1[j] + y_false_class1[j]) * 100)
                plt.text(i+0.32, y_true_class1[j]-3, texto, verticalalignment='bottom', fontsize='15')
                
                texto = '{:.2f}%'.format(y_false_class1[j] / (y_true_class1[j] + y_false_class1[j]) * 100)
                plt.text(i+0.53, y_false_class1[j]-3, texto, verticalalignment='bottom', fontsize='15')
            except:
                pass
            finally:
                j += 1



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
from sklearn.metrics import accuracy_score, confusion_matrix
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt

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
# Selecionar os modelos alterando a variavel use_models
all_models = {1: 'naive bayes', 2: 'arvore', 3: 'forest',
              4: 'knn', 5: 'regressao', 6: 'svm', 7: 'rna'}
use_models = [1, 2, 3, 4, 5, 6]
nome_arquivo = 'resultados.csv'


# =====   Coleta dos resultados e Criação do arquivo csv   ===== #
# Criacao do dataframe que armazenara os resultados
seeds = 30
result_dataframe = DataFrame(list(range(seeds)), columns=['seed'])
average_result = DataFrame()

model_result = []
model_matrix_result = []
model_name = []

# Coletar os resultados associando-os aos respectivos modelos
# e Armazena-los em disco em formato csv
executa_avaliacao(seeds, use_models)
#executa_avaliacao(seeds, [3]) # Testar 1 unico modelo


# =====   Exibição dos Resultados Finais   ===== #
# Visualização Grafica das Precisoes
view_accuracy()

# Visualização do DataFrame das Precisoes
print(average_result)

# Visualização das Matrizes de Confusão
# OBS: Funciona apenas para base de dados com 2 valores de classe
# Visualiza a media das matrizes de confusão e realiza uma comparação gráfica
for matriz, name in zip(model_matrix_result, model_name):
    print("{}:\n {}".format(name, matriz))

view_confusion_matrix(classe0='bom', classe1='ruim', detail_models=[6, 3])
# detalha os modelos randomforest(3) e svm(6), numeração relativa a variavel all_models