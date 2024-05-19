import numpy as np

# Definindo o Perceptron (adaptado para o Backpropagation)
class PerceptronBackpropagation(object):
    
    # Definindo o padrão do Perceptron
    def __init__(self, taxa_aprendizado, epoca, num_neuronios_oculta):

        self.taxa_aprendizado = taxa_aprendizado
        self.epoca = epoca
        self.num_neuronios_oculta = num_neuronios_oculta

    # Função de ativação usada: sigmoide
    def _sigmoide(self, x):
        return 1 / (1 + np.exp(-x))

    # Calculando a derivada da função de ativação (para ser usada no gradiente descendente)
    def _sigmoide_derivada(self, x):
        return x * (1 - x)

    # Inicializando os pesos
    def _inicializa_pesos(self, num_features):
    
        self.pesos_oculta = np.random.uniform(-1, 1, (num_features, self.num_neuronios_oculta))
        self.pesos_saida = np.random.uniform(-1, 1, (self.num_neuronios_oculta, 1))

    # Calculando as saídas das camadas usando a função de ativação
    def predict(self, X):
    
        camada_oculta = self._sigmoide(np.dot(X, self.pesos_oculta))
        return self._sigmoide(np.dot(camada_oculta, self.pesos_saida))

    # Realizando o treinamento do Perceptron usando o Backpropagation
    def fit(self, X, y):

        self._inicializa_pesos(X.shape[1])

        for _ in range(self.epoca):
            
            camada_oculta = self._sigmoide(np.dot(X, self.pesos_oculta))
            output = self._sigmoide(np.dot(camada_oculta, self.pesos_saida))

            erro_saida = y - output
            erro_camada_oculta = erro_saida.dot(self.pesos_saida.T) * self._sigmoide_derivada(camada_oculta)

            self.pesos_saida += camada_oculta.T.dot(erro_saida) * self.taxa_aprendizado
            self.pesos_oculta += X.T.dot(erro_camada_oculta) * self.taxa_aprendizado

        return self
    
# Dados de 2 entradas
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def funcao_or(X):
    return np.array([[0], [1], [1], [1]])

def funcao_and(X):
    return np.array([[0], [0], [0], [1]])

def funcao_xor(X):
    return np.array([[0], [1], [1], [0]])

parar = 'fim'

while True:
    funcao_booleana = input("Digite a função booleana desejada (and, or, xor) ou fim para sair: ")
    if funcao_booleana == 'or':
        y = funcao_or(X)
    elif funcao_booleana == 'and':
        y = funcao_and(X)
    elif funcao_booleana == 'xor':
        y = funcao_xor(X)

    if funcao_booleana == parar:
        break

    else: 
        numero_entradas = int(input("Digite o número de entradas desejado: "))
        perceptron = PerceptronBackpropagation(taxa_aprendizado = 0.1, epoca = 10000, num_neuronios_oculta = 4)

        perceptron.fit(X, y)

        print("Saída predita para a função:")
        print(perceptron.predict(X))

        limiar_decisao = 0.4
        saidas_previstas_binarias = perceptron.predict(X) > limiar_decisao
        saidas_previstas_binarias = saidas_previstas_binarias.astype(int)
        print("Saída binária predita para a função: ")
        print(saidas_previstas_binarias.flatten())
