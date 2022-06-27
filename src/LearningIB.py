#faire une classe "cerveau" ou l'on choisi le type de cerveau (qtable, deep, deepqtable, machine learning)

from NeuralNetwork.NeuralNetwork import *
from NeuralNetwork.__Neuron import *
from NeuralNetwork.__ActivationFunction import *

from ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork import *
from ConvolutionalNeuralNetwork.__ActivationFunction import *

BRAIN_TYPE_NEURAL_NETWORK = 0
BRAIN_TYPE_CONVOLUTIONAL_NEURAL_NETWORK = 1
# BRAIN_TYPE_QLEARNING = 2
# BRAIN_TYPE_DEEP_QLEARNING = 3

class IBBrain:
    __brainType = None                                          #Type de cerveau (BRAIN_TYPE_####)
    __neuralNetwork = None                                      #Reseau de neurone
    __convolutionalNeuralNetwork = None                         #Reseau de neurone de convolution
    __qTable = None                                             #QTable
    __deepQTable = None                                         #Deep QLearning

    def __init__(self, brainType: int) -> None:
        """
            Cerveau pour IA
            Params:
                brainType : Type de cerveau (BRAIN_TYPE_####)
        """
        self.__brainType = brainType
        return

    def LoadBrain(self, fileName: str) -> None:
        """
            Chargement du cerveau depuis un fichier
            Params:
                fileName : Nom du fichier
        """
        if self.__brainType == BRAIN_TYPE_NEURAL_NETWORK:           #Si reseau de neurone
            self.__neuralNetwork = NeuralNetwork()                      #Création de l'objet
            self.__neuralNetwork.LoadNetwork(fileName)                  #Chargement
        if self.__brainType == BRAIN_TYPE_CONVOLUTIONAL_NEURAL_NETWORK:#Si réseau de neurone a convolution
            self.__convolutionalNeuralNetwork = ConvolutionalNeuralNetwork(0, 0, NEURON_OUTPUT_CELL, FUNCTION_SIGMOID)#Creation du cerveau (ici avec des valeurs vide car elles seront modifier)
            self.__convolutionalNeuralNetwork.LoadNetwork(fileName)     #Chargement du reseau de neurone
        return
    def SaveBrain(self, fileName: str) -> None:
        """
            Sauvegarde du cerveau dans un fichier
            Params:
                fileName : Nom du fichier
        """        
        if self.__brainType == BRAIN_TYPE_NEURAL_NETWORK: self.__neuralNetwork.SaveNetwork(fileName)#Si reseau de neurone => Sauvegarde du reseau
        if self.__brainType == BRAIN_TYPE_CONVOLUTIONAL_NEURAL_NETWORK: self.__convolutionalNeuralNetwork.SaveNetwork(fileName)#Si CNN => Sauvegarde du reseau
        return

    def SetNeuralNetwork(self, neuralNetwork: NeuralNetwork) -> None:
        """
            Definition du reseau de neurone
            Params:
                neuralNetwork : Réseau de neurone déjà paramétré
        """
        self.__neuralNetwork = neuralNetwork
        return
    def TrainNeuralNetwork(self, dataset: list, expected: list, nbEpoch: int, learningRate: float = 0.5, acceleration: bool = False) -> None:
        """
            Entrainement du reseau de neurone
            Params:
                dataset : Liste de liste d'entrée
                expected : Liste de liste de sortie attendu
                nbEpoch : Nombre d'entrainement
                learningRate : Taux d'apprentissage
                acceleration : Accélération materiel
        """
        self.__neuralNetwork.Train(dataset, expected, nbEpoch, learningRate, acceleration)
        return
    def PredictNeuralNetwork(self, inputs: list, acceleration: bool = False) -> list:
        """
            Prédiction du reseau de neurone
            Params:
                inputs : Liste de liste d'entrée
                acceleration : Accélération materiel
            Return:
                Liste des sorties
        """
        return self.__neuralNetwork.Predict(inputs, acceleration)

    def SetConvolutionalNeuralNetwork(self, convolutionalNeuralNetwork: ConvolutionalNeuralNetwork) -> None:
        """
            Définition du reseau de neurone CNN
            Params:
                convolutionalNeuralNetwork : Reseau de neurone CNN
        """
        self.__convolutionalNeuralNetwork = convolutionalNeuralNetwork
        return
    def TrainConvolutionalNeuralNetwork(self, dataset: list, expected: list, nbEpoch: int, learningRate: float = 0.5, acceleration: bool = False) -> None:
        """
            Entrainement du reseau de neurone CNN
            Params:
                dataset : Liste de liste d'entrée
                expected : Liste de liste de sortie attendu
                nbEpoch : Nombre d'entrainement
                learningRate : Taux d'apprentissage
                acceleration : Accélération materiel
        """
        self.__convolutionalNeuralNetwork.Train(dataset, expected, nbEpoch, learningRate, acceleration)
        return
    def PredictConvolutionalNeuralNetwork(self, inputs: list) -> None:
        """            
            Prédiction du reseau de neurone CNN
            Params:
                inputs : Liste de liste d'entrée
            Return:
                Liste des sorties
        """
        return self.__convolutionalNeuralNetwork.Predict(inputs)


from timeit import default_timer as timer
#ce qu'on veux c'est que les deux centre soit 1
dataset = [
    [0,1,1,0],
    [1,1,0,1],
    [1,1,1,1],
    [0,0,0,0],
    [1,0,1,0],
    [0,1,0,1],
    [0,1,1,1],
    [1,1,1,0],
    [1,0,0,0],
    [1,0,1,1],
    [0,0,1,1],
    [0,1,1,0],
    [1,1,0,1],
    [0,1,1,1],
    [0,1,0,1],
    [1,0,1,0],    
    [0,0,0,1],
    [0,0,1,1],
    [0,1,1,1],
    [1,0,1,0],
    [1,1,1,1]
]
expected = [
    [1],
    [0],
    [1],
    [0],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [0],
    [1],
    [0],
    [1],
    [0],
    [0],
    [0],
    [0],
    [1],
    [0],
    [1]
]

brainType = int(input("0) Neural Network\n1) CNN\nBrain type : "))

if brainType == BRAIN_TYPE_NEURAL_NETWORK:
    #----NEURAL NETWORK----#
    brain = IBBrain(BRAIN_TYPE_NEURAL_NETWORK)

    neuralNetwork = NeuralNetwork()

    inputs = []
    inputs.append(neuralNetwork.AddNeuron(NEURON_INPUT_CELL, NETWORK_POSITION_INPUT))
    inputs.append(neuralNetwork.AddNeuron(NEURON_INPUT_CELL, NETWORK_POSITION_INPUT))
    inputs.append(neuralNetwork.AddNeuron(NEURON_INPUT_CELL, NETWORK_POSITION_INPUT))
    inputs.append(neuralNetwork.AddNeuron(NEURON_INPUT_CELL, NETWORK_POSITION_INPUT))

    hiddenLayer1 = []
    hiddenLayer1.append(neuralNetwork.AddNeuron(NEURON_HIDDEN_CELL, NETWORK_POSITION_HIDDEN, FUNCTION_SIGMOID, inputs))
    hiddenLayer1.append(neuralNetwork.AddNeuron(NEURON_HIDDEN_CELL, NETWORK_POSITION_HIDDEN, FUNCTION_SIGMOID, inputs))
    hiddenLayer1.append(neuralNetwork.AddNeuron(NEURON_HIDDEN_CELL, NETWORK_POSITION_HIDDEN, FUNCTION_SIGMOID, inputs))
    hiddenLayer1.append(neuralNetwork.AddNeuron(NEURON_HIDDEN_CELL, NETWORK_POSITION_HIDDEN, FUNCTION_SIGMOID, inputs))
    hiddenLayer1.append(neuralNetwork.AddNeuron(NEURON_HIDDEN_CELL, NETWORK_POSITION_HIDDEN, FUNCTION_SIGMOID, inputs))
    hiddenLayer1.append(neuralNetwork.AddNeuron(NEURON_HIDDEN_CELL, NETWORK_POSITION_HIDDEN, FUNCTION_SIGMOID, inputs))
    hiddenLayer1.append(neuralNetwork.AddNeuron(NEURON_HIDDEN_CELL, NETWORK_POSITION_HIDDEN, FUNCTION_SIGMOID, inputs))
    hiddenLayer1.append(neuralNetwork.AddNeuron(NEURON_HIDDEN_CELL, NETWORK_POSITION_HIDDEN, FUNCTION_SIGMOID, inputs))

    outputs = []
    outputs.append(neuralNetwork.AddNeuron(NEURON_OUTPUT_CELL,NETWORK_POSITION_OUTPUT,FUNCTION_SIGMOID, hiddenLayer1))

    brain.SetNeuralNetwork(neuralNetwork)

    for inputs in dataset:
        print(brain.PredictNeuralNetwork(inputs, True))

    start = timer()
    brain.TrainNeuralNetwork(dataset, expected, 10000, 0.5, True)
    print(timer()-start,"s")

    for inputs in dataset:
        print(brain.PredictNeuralNetwork(inputs, True))
    #----#############----#
if brainType == BRAIN_TYPE_CONVOLUTIONAL_NEURAL_NETWORK:
    #----CNN----#
    brain = IBBrain(BRAIN_TYPE_CONVOLUTIONAL_NEURAL_NETWORK)
    cnn = ConvolutionalNeuralNetwork(4, 1, NEURON_OUTPUT_CELL, FUNCTION_SIGMOID)
    cnn.AddLayer(8, NEURON_HIDDEN_CELL, FUNCTION_SIGMOID)
    brain.SetConvolutionalNeuralNetwork(cnn)

    print("#----CPU----#")
    tauxReussite = 0
    for x in range(len(dataset)):
        prediction = brain.PredictConvolutionalNeuralNetwork(dataset[x])[0]
        print("Receive :",round(prediction), "Expected :", expected[x],"     ->",prediction)
        if round(prediction) == expected[x][0]: tauxReussite += 1
    print("Le taux de reussite est de",(tauxReussite/len(dataset))*100,"%")

    start = timer()
    brain.TrainConvolutionalNeuralNetwork(dataset, expected, 1000, 0.5, False)
    print(timer()-start,"s")

    tauxReussite = 0
    for x in range(len(dataset)):
        prediction = brain.PredictConvolutionalNeuralNetwork(dataset[x])[0]
        print("Receive :",round(prediction), "Expected :", expected[x],"     ->",prediction)
        if round(prediction) == expected[x][0]: tauxReussite += 1
    print("Le taux de reussite est de",(tauxReussite/len(dataset))*100,"%")

    #----###----#


input("ok")