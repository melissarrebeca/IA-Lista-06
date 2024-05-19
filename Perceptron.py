import numpy as np

# Definindo o Perceptron
class Perceptron(object):
    
    # Definindo o padrão do Perceptron
    def __init__(self, taxa_aprendizado, epoca):
        
        self.taxa_aprendizado = taxa_aprendizado
        self.epoca = epoca

    # Soma ponderada dos pesos
    def soma_ponderada(self, X):
        return np.dot(X, self.pesos_[1:]) + self.pesos_[0]
    
    # Função de ativação
    def predict(self, X):
        return np.where(self.soma_ponderada(X) >= 0.0, 1, -1)

    # Treinando o Perceptron
    def fit(self, X, y):
        
        # Inicializando os pesos e a lista de erros
        self.pesos_ = np.zeros(1 + X.shape[1])
        self.erros_ = []

        # Treinando o modelo de acordo com o número de épocas
        for _ in range(self.epoca):
            erro = 0

            # Para cada entrada:
            for vetor_treinamento, saida_esperada in zip(X, y):

                # 1. Calcula y^ (Valor encontrado)
                saida_encontrada = self.predict(vetor_treinamento)

                # 2. Atualiza taxa de aprendizado
                nova_taxa_aprendizado = self.taxa_aprendizado * (saida_esperada - saida_encontrada)

                # 3. Realiza o update nos pesos
                self.pesos_[1:] = self.pesos_[1:] + nova_taxa_aprendizado * vetor_treinamento
                #print(self.w_[1:])

                # 4. Realiza o update do bias
                self.pesos_[0] = self.pesos_[0] + nova_taxa_aprendizado

                erro += int(nova_taxa_aprendizado != 0.0)

            self.erros_.append(erro)

        return self

def gera_dados_booleanos(numero_entradas, funcao_booleana):

    if funcao_booleana == 'and':

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, -1, -1, 1])

    elif funcao_booleana == 'or':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, 1, 1, 1])

    elif funcao_booleana == 'xor':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, 1, 1, -1])

    if numero_entradas > 2:
        X = np.vstack([X]*int(2**(numero_entradas - 2)))

    return X, y

def usa_perceptron(numero_entradas, funcao_booleana):

    X, y = gera_dados_booleanos(numero_entradas, funcao_booleana)

    perceptron = Perceptron(taxa_aprendizado = 0.1, epoca = 5)
    perceptron.fit(X, y)

    print(f"Perceptron treinado com a função {funcao_booleana} e {numero_entradas} entradas: ")
    print("Pesos finais: ", perceptron.pesos_)
    print("Erros por época: ", perceptron.erros_)

    saida_convertida = np.where(perceptron.predict(X) == -1, 0, perceptron.predict(X))
    print("Saída binária predita para a função: ",saida_convertida)
    print()

parar = 'fim'
while True:
    funcao_booleana = input("Digite a função booleana desejada (and, or, xor) ou fim para sair: ")

    if funcao_booleana == parar:
        break
    else: 
        numero_entradas = int(input("Digite o número de entradas desejado: "))
        usa_perceptron(numero_entradas, funcao_booleana)

