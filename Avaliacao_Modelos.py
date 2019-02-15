# Processo de Avaliação de modelos utilizando Validação Cruzada

# Neste ponto deve-se ter definido os atributos previsores e a classe
# E já ter realizados todos os pre processamentos de dados necessários



########## ########## ########## ##########
# Funções Auxiliares
########## ########## ########## ##########
def create_test_model(classificador, index_train, index_test):
    # Cria e Testa o Modelo escolhido
    if classificador == 'naive bayes':
        from sklearn.naive_bayes import GaussianNB
        modelo = GaussianNB()
    elif classificador == 'arvore':
        from sklearn.tree import DecisionTreeClassifier
        modelo = DecisionTreeClassifier()
    elif classificador == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        modelo = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
    elif classificador == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        modelo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    elif classificador == 'regressao':
        from sklearn.linear_model import LogisticRegression
        modelo = LogisticRegression()
    elif classificador == 'svm':
        from sklearn.svm import SVC
        modelo = SVC(kernel='rbf', random_state=1, C=2.0)
    elif classificador == 'rede neural':
        from sklearn.neural_network import MLPClassifier
        modelo = MLPClassifier(max_iter=1000, tol=0.000001, solver='adam',
                           hidden_layer_sizes=(100), activation='relu',
                           batch_size=200, learning_rate=0.001)
    else:
        raise NameError ('Modelo escolhido nao esta na base de dados')
    
    modelo.fit(previsores[index_train], classe[index_train])
    previsoes = modelo.predict(previsores[index_test])

    return previsoes


def avalia_modelos(previsores, classe, classificador, n_seed=30, n_folds=10):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from numpy import zeros, array

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

    ##### ########## ########## ########## ########## ########## #####
    ##### ########## ########## ########## ########## ########## #####



########## ########## ########## ##########
# Avaliação com Validação Cruzada
########## ########## ########## ##########

# =====   Carregamento dos atributos previsores e classe   ===== #
previsores = []
classe = []

    
# =====   Seleção dos modelos a utilizar   ===== #
# Selecionar os modelos alterando a variavel use_models
all_models = {1: 'naive bayes', 2: 'arvore', 3: 'forest',
              4: 'knn', 5: 'regressao', 6: 'svm', 7: 'rede neural'}
use_models = [1, 2, 4]
# para testar 1 unico modelo, descomente linha abaixo
# resultados = avalia_modelos(previsores, classe, classificador=all_models[1])

# =====   Coleta dos resultados   ===== #
# Coletar os resultados associando-os aos respectivos modelos
resultados = []
model_name = []
for modelo in use_models:
    model_name.append(all_models[modelo].title())
    resultados.append(avalia_modelos(previsores, classe, classificador=all_models[modelo]))


# =====   Criação do arquivo csv com os resultados   ===== #
from pandas import DataFrame
arquivo = []
# Percorre o resultado de cada modelo para cada seed
for n in range(len(resultados[0])):
    result_seed = []
    for i in range(len(resultados)):
        result_seed.append(resultados[i][n])
    arquivo.append(result_seed)
# Transformação para DataFrame para criacao do arquivo csv
arquivo = DataFrame(arquivo, columns=model_name)
arquivo.to_csv('resultados.csv')
