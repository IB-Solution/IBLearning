#OPTIMISATION GPU
# Cout moyen
# ASYNCHRONE

import random
import json
import numpy as np
from numba import jit, njit
from .__Neuron import *

NETWORK_POSITION_INPUT = 0
NETWORK_POSITION_HIDDEN = 1
NETWORK_POSITION_OUTPUT = 2

class NeuralNetwork:
    neuronList = None                                   #Dictionnaire des neurones du reseau {"numeroNeurone":Neuron}
    inputList = None                                    #Liste des neurones d'entrée du reseau
    hiddenList = None                                   #Liste des neurones caché
    outputList = None                                   #Liste des neurones de sortie du reseau

    def __init__(self) -> None:
        self.neuronList = {}
        self.inputList = []
        self.hiddenList = []
        self.outputList = []
        
        return

    def AddNeuron(self, neuronType: int, neuronPosition: int, activationFunction: int = None, inputNeuronList: list = [], alpha: float = 1) -> str:
        """
            Ajoute un neurone au réseau
            Params:
                neuronType : Type de neurone NEURON_###_CELL
                neuronPosition : Position du neurone dans le reseau (si entrée, sortie ou caché) NETWORK_POSITION_####
                activationFunction : Fonction d'activation FUNCTION_#####
                inputNeuronList : Liste des noms des neurones d'entrée
                alpha : Alpha du calcul de l'activation
            Return:
                Nom du neurone créé
        """
        neuronName = self.__NameGenerator()                                                 #Generation d'un nom unique
        self.neuronList[neuronName] = Neuron(neuronName= neuronName, neuronType= neuronType, activationFunction= activationFunction, alpha= alpha)#Generation du neurone
        if neuronPosition == NETWORK_POSITION_INPUT:                                        #Si le neurone est une entrée
            self.inputList.append(neuronName)                                                   #On ajoute l'entrée
        if neuronPosition == NETWORK_POSITION_HIDDEN:                                       #Si le neurone est caché
            self.hiddenList.append(neuronName)                                                  #Ajout du neurone dans la liste
            for inputNeuronName in inputNeuronList:                                             #Pour chaque entrée du neurone
                weight = self.neuronList[neuronName].AddInput(inputNeuronName)                      #Ajout du neurone d'entrée
                self.neuronList[inputNeuronName].AddOutput(neuronName, weight)                      #Ajout du poids de sortie pour finalisé la liaison
        if neuronPosition == NETWORK_POSITION_OUTPUT:                                       #Si le neurone est une sortie
            self.outputList.append(neuronName)                                                  #Ajout du neurone dans la liste
            for inputNeuronName in inputNeuronList:                                             #Pour chaque entrée du neurone
                weight = self.neuronList[neuronName].AddInput(inputNeuronName)                      #Ajout du neurone d'entrée
                self.neuronList[inputNeuronName].AddOutput(neuronName, weight)                      #Ajout du poids de sortie pour finalisé la liaison

        return neuronName                                                                   #Retour du nom du neurone


    def Train(self, dataset: list, expected: list, nbEpoch: int, learningRate: float, acceleration: bool = False, display: bool = False) -> float:
        """
            Entrainement du reseau de neurone
            Params:
                dataset : Liste de liste d'entrées
                expected : Liste de liste de sortie attendue en liaison avec la dataset
                nbEpoch : Nombre d'entrainement sur toute la dataset
                learningRate : Taux d'apprentissage (0 = aucun apprentissage, 1 = Trop d'apprentissage)
                acceleration : Accélération materiel
                display : Si on affiche
            Return:
                Taux d'erreur du reseau
        """
        cout = 0                                        #Taux d'erreur
        for epoch in range(nbEpoch):                    #Pour chaque epoch
            cout = 0                                        #Remise a 0 du taux d'erreur
            for x, inputs in enumerate(dataset):            #Pour chaque entrée de la dataset
                self.FeedForward(inputs, acceleration)          #Lancement du calcul
                result = self.__GetResult()                     #Récupération de la sortie
                cout += sum([(expected[x][i] - result[i])**2 for i in range(len(expected[x]))])#Calcul du taux d'erreur
                self.BackPropagation(expected[x], acceleration) #Lancement du calcul du taux d'erreur
                self.UpdateWeights(learningRate)                #Mise a jour des poids
            if display: print("====="+str(epoch)+"/"+str(nbEpoch)+"=====\nCout : \n"+str(cout)+"\n=====###=====")
        return cout
    def Predict(self, inputs: list, acceleration: bool = False) -> list:
        """
            Prédiction du reseau de neurone
            Params:
                inputs : Liste des entrées
                acceleration : Acceleration materiel
            Return:
                Liste des sortie
        """
        self.FeedForward(inputs, acceleration)          #Sinon => Calcul du reseau
        return self.__GetResult()                       #Retour du resultat


    def FeedForward(self, inputs: list, acceleration: bool = False) -> None:
        """
            Propagation des valeurs et calcul des neurones
            Params:
                inputs : Données d'entrées du reseau de neurones
                acceleration : Si utilisation de l'accélération matériel
        """
        leftNeuron = list(self.neuronList.keys())       #Liste des nom de neurones non calculé
        #RESET
        for neuronName in leftNeuron:                   #Pour chaque neurone du reseau
            self.neuronList[neuronName].done = False        #On le definie comme non calculé

        #INPUTS
        for x, neuronName in enumerate(self.inputList): #Pour chaque neurone d'entrée
            self.neuronList[neuronName].value = inputs[x]   #On définie la valeur d'entrée
            self.neuronList[neuronName].done = True         #Neuron calculé
            del leftNeuron[leftNeuron.index(neuronName)]    #Suppression du neurone calculé

        #HIDDEN & OUTPUTS
        if acceleration:                                #Si acceleration materiel
            while len(leftNeuron) != 0:                     #Tant que tout les neurones ne sont pas calculé
                for neuronName in leftNeuron:                   #Pour chaque neurone non calculé 
                    if self.neuronList[neuronName].CheckInputDependance(self.neuronList):#Si le neurone actuel à des dépendances déjà calculé
                        self.neuronList[neuronName].FeedForwardGPU(self.neuronList)#Calcul du neurone
                        del leftNeuron[leftNeuron.index(neuronName)]#Suppression du neurone calculé
        else:                                           #Sinon
            while len(leftNeuron) != 0:                     #Tant que tout les neurones ne sont pas calculé
                for neuronName in leftNeuron:                   #Pour chaque neurone non calculé 
                    if self.neuronList[neuronName].CheckInputDependance(self.neuronList):#Si le neurone actuel à des dépendances déjà calculé
                        self.neuronList[neuronName].FeedForward(self.neuronList)#Calcul du neurone
                        del leftNeuron[leftNeuron.index(neuronName)]#Suppression du neurone calculé
        return
    def BackPropagation(self, expected: list, acceleration: bool = False) -> None:
        """
            Calcul du taux d'erreur de chaque neurone
            Params:
                expected : Liste des données attendue de chaque neurone de sortie
                acceleration : Si utilisation de l'accélération matériel
        """
        leftNeuron = list(self.neuronList.keys())       #Liste des nom de neurones non calculé
        #RESET
        for neuronName in leftNeuron:                   #Pour chaque neurone du reseau
            self.neuronList[neuronName].done = False        #On le definie comme non calculé
            if neuronName in self.inputList:                #Si le neurone est un neurone d'entrée
                del leftNeuron[leftNeuron.index(neuronName)]    #On retire le neurone des neurones a calculer

        #OUTPUTS
        for x, neuronName in enumerate(self.outputList):#Pour chaque neurone de sortie
            self.neuronList[neuronName].BackProgation(self.neuronList, expected[x])#Calcul du taux d'erreur du neurone de sortie
            del leftNeuron[leftNeuron.index(neuronName)]    #Suppression du neurone calculé

        #HIDDEN
        if acceleration:                                #Si accélération materiel
            while len(leftNeuron) != 0:                     #Tant que tout les neurones ne sont pas calculé
                for neuronName in leftNeuron:                   #Pour chaque neurone non calculé 
                    if self.neuronList[neuronName].CheckOutputDependance(self.neuronList):#Si le neurone actuel à des dépendances déjà calculé
                        self.neuronList[neuronName].BackProgationGPU(self.neuronList)#Calcul du taux d'erreur du neurone
                        del leftNeuron[leftNeuron.index(neuronName)]
        else:                                           #sinon
            while len(leftNeuron) != 0:                     #Tant que tout les neurones ne sont pas calculé
                for neuronName in leftNeuron:                   #Pour chaque neurone non calculé 
                    if self.neuronList[neuronName].CheckOutputDependance(self.neuronList):#Si le neurone actuel à des dépendances déjà calculé
                        self.neuronList[neuronName].BackProgation(self.neuronList)#Calcul du taux d'erreur du neurone
                        del leftNeuron[leftNeuron.index(neuronName)]
        return
    def UpdateWeights(self, learningRate: float) -> None:
        """
            Mise a jour du poids des liaisons entre les neurones
            Params:
                learningRate : Taux d'apprentissage (0 = aucun apprentissage, 1 = Trop d'apprentissage)
        """
        leftNeuron = list(self.neuronList.keys())       #Liste des nom de neurones non calculé
        #RESET & INPUTS
        for neuronName in leftNeuron:                   #Pour chaque neurone du reseau
            if neuronName in self.inputList:                #Si le neurone est une entrée
                self.neuronList[neuronName].done = True         #Neuron calculé
                del leftNeuron[leftNeuron.index(neuronName)]    #Suppression du neurone calculé
            else:                                           #Sinon
                self.neuronList[neuronName].done = False        #On le definie comme non calculé

       #HIDDEN & OUTPUT
        while len(leftNeuron) != 0:                     #Tant que tout les neurones ne sont pas calculé
            for neuronName in leftNeuron:                   #Pour chaque neurone non calculé 
                if self.neuronList[neuronName].CheckInputDependance(self.neuronList):#Si le neurone actuel à des dépendances déjà calculé
                    self.neuronList[neuronName].UpdateWeight(self.neuronList, learningRate)#Mise a jour du neurone
                    del leftNeuron[leftNeuron.index(neuronName)]
        return



    def SaveNetwork(self, fileName: str) -> None:
        """
            Sauvegarde du reseau de neurone dans un fichier
            Params:
                fileName : Nom du fichier de destination
        """
        output = {}                                     #JSON du réseau
        
        output["inputList"] = self.inputList            #Liste d'entrée du reseau
        output["hiddenList"] = self.hiddenList          #Liste caché du reseau
        output["outputList"] = self.outputList          #Liste de sortie du reseau

        output["neuronList"] = {}                       #Liste des neurones du reseau
        for neuronName in self.neuronList:              #Pour chaque neurone de la liste
            output["neuronList"][neuronName] = {}          #Données du neurone
            output["neuronList"][neuronName]["neuronType"] = self.neuronList[neuronName].neuronType#Type du neurone
            output["neuronList"][neuronName]["inputList"] = self.neuronList[neuronName].inputList#Liste d'entrée
            output["neuronList"][neuronName]["outputList"] = self.neuronList[neuronName].outputList#Liste de sortie
            output["neuronList"][neuronName]["activationFunction"] = self.neuronList[neuronName].activationFunction#Fonction d'activation
            output["neuronList"][neuronName]["biais"] = self.neuronList[neuronName].biais#Taux d'activation du neurone
            output["neuronList"][neuronName]["alpha"] = self.neuronList[neuronName].alpha#Pour fonction d'activation
        
        outputFile = open(fileName, "w")
        json.dump(output, outputFile, indent = 4)
        outputFile.close()
        return
    def LoadNetwork(self, fileName: str) -> None:
        """
            Chargement du reseau de neurone depuis un fichier
            Params:
                fileName : Nom du fichier source
        """
        inputFile = open(fileName)                      #Ouverture du fichier
        inputData = json.load(inputFile)                #Chargement du json
        inputFile.close()                               #Fermeture du fichier

        self.inputList = inputData["inputList"]         #Liste des entrées
        self.hiddenList = inputData["hiddenList"]       #Liste caché
        self.outputList = inputData["outputList"]       #Liste des sorties
        
        for neuronName in inputData["neuronList"]:      #Pour chaque neurone
            self.neuronList[neuronName] = Neuron(neuronName, inputData["neuronList"][neuronName]["neuronType"], inputData["neuronList"][neuronName]["activationFunction"], inputData["neuronList"][neuronName]["alpha"])#Création du neurone original
            self.neuronList[neuronName].biais = inputData["neuronList"][neuronName]["biais"]
            self.neuronList[neuronName].inputList = inputData["neuronList"][neuronName]["inputList"]
            self.neuronList[neuronName].outputList = inputData["neuronList"][neuronName]["outputList"]

        return


    def __GetResult(self) -> list:
        """
            Récupération de la sortie du reseau
            Return:
                Liste des valeurs de sorties
        """
        return [self.neuronList[i].value for i in self.outputList]

    def __NameGenerator(self) -> str:
        """
            Genere le nom d'un neurone qui n'est pas encore dans le reseau
            Return:
                Nom du neurone
        """
        neuronName = "0"
        while neuronName in self.neuronList.keys():
            neuronName = str(random.randint(1,9999999))
        return neuronName



# def TrainGPU(fileName: str ,dataset: np.ndarray, expected: np.ndarray, nbEpoch: int, learningRate: float) -> None:
#     """
#         Entrainement du reseau
#         Params:
#             fileName : Nom du fichier du reseau
#             dataset : Liste des entrées
#             expected : Liste des sorties attendue
#             nbEpoch : Nombre d'entrainement
#             learningRate : Taux d'apprentissage
#     """
#     #Pour ce repérer, tout est en fonction de l'index
#     neuronIndex = {}                                #{"neuronName": index}
#     neuronsName = {}                                #{"neuronID": neuronName}

#     inputFile = open(fileName)                      #Ouverture du fichier
#     inputData = json.load(inputFile)                #Chargement du json
#     inputFile.close()                               #Fermeture du fichier

#     nbNeuron = len(inputData["neuronList"].keys())  #Nombre de neurone du reseau
#     neuronType = np.zeros(nbNeuron, dtype= np.int)  #Type du neurone
#     activationFunction = np.zeros(nbNeuron, dtype= np.int)#Liste des activations [activationNeuron1, activationNeuron2, activationNeuron3, ...]
#     alpha = np.zeros(nbNeuron, np.float64)          #Alpha
#     value = np.zeros(nbNeuron, dtype= np.float64)   #Liste des valeur des neurone [valueNeuron1, valueNeuron2, valueNeuron3, ...]
#     biais = np.zeros(nbNeuron, dtype= np.float64)   #Liste des biais des neurone [biaisNeuron1, biaisNeuron2, biaisNeuron3, ...]
#     cout = np.zeros(nbNeuron, dtype= np.float64)    #Liste des cout des neurone [coutNeuron1, coutNeuron2, coutNeuron3, ...]
#     delta = np.zeros(nbNeuron, dtype= np.float64)   #Liste des taux d'erreur des neurone [deltaNeuron1, deltaNeuron2, deltaNeuron3, ...]
    
#     done = np.zeros(nbNeuron, dtype= np.bool)       #Liste de si les neurones on était calculé [neuron1, neuron2, neuron3, ...]

#     nbNeuronInput = np.zeros(nbNeuron, np.int)              #[nbInputNeuron1,nbInputNeuron2,nbInputNeuron3,...]
#     inputNeuronList = np.zeros((nbNeuron,nbNeuron,2), np.float64)#liste des entrées d'un neurone [ [ [indexNeuron1->1, poids], [indexNeuron2->1, poids] ],] On prepare une taille maximal au cas où
#     nbNeuronOutput = np.zeros(nbNeuron, np.int)             #[nbOutputNeuron1,nbOutputNeuron2,nbOutputNeuron3,...]
#     outputNeuronList = np.zeros((nbNeuron,nbNeuron,2), np.float64)#On prepare une taille maximal au cas où (idem que le précédent)

#     nbInputs = 0
#     networkInputs = np.zeros(nbNeuron, np.int)      #Index des neurones d'entrée
#     nbHidden = 0
#     networkHidden = np.zeros(nbNeuron, np.int)      #Index des neurones de sortie
#     nbOutput = 0
#     networkOutputs = np.zeros(nbNeuron, np.int)      #Index des neurones de sortie

#     #Creation de l'index pour associer le nom Text en index
#     for x, neuronName in enumerate(inputData["neuronList"].keys()):#Pour chaque neurone
#         neuronIndex[neuronName] = x
#         neuronsName[x] = neuronName


#     for neuronName in neuronIndex.keys():           #Pour chaque neurone
#         index = neuronIndex[neuronName]

#         neuronType[index]           =       inputData["neuronList"][neuronName]["neuronType"]
#         if inputData["neuronList"][neuronName]["activationFunction"] == None : activationFunction[index]     =       -1
#         else : activationFunction[index]     =       inputData["neuronList"][neuronName]["activationFunction"]
#         biais[index]                =       inputData["neuronList"][neuronName]["biais"]
#         alpha[index]                =       inputData["neuronList"][neuronName]["alpha"]

#         #Entrée du neurone
#         for inputNeuronName in inputData["neuronList"][neuronName]["inputList"].keys():
#             inputNeuronList[ index ][ nbNeuronInput[index] ][ 0 ] = neuronIndex[inputNeuronName]
#             inputNeuronList[ index ][ nbNeuronInput[index] ][ 1 ] = inputData["neuronList"][neuronName]["inputList"][inputNeuronName]
#             nbNeuronInput[index] += 1

#         #Sortie du neurone
#         for outputNeuronName in inputData["neuronList"][neuronName]["outputList"].keys():
#             outputNeuronList[ index ][ nbNeuronOutput[index] ][ 0 ] = neuronIndex[outputNeuronName]
#             outputNeuronList[ index ][ nbNeuronOutput[index] ][ 1 ] = inputData["neuronList"][neuronName]["outputList"][outputNeuronName]
#             nbNeuronOutput[index] += 1

#         #Entrée du réseau
#         if neuronName in inputData["inputList"]: 
#             networkInputs[nbInputs] =       index
#             nbInputs += 1
#         #Caché du reseua
#         if neuronName in inputData["hiddenList"]: 
#             networkHidden[nbHidden] =       index
#             nbHidden += 1
#         #Sortie du réseau
#         if neuronName in inputData["outputList"]: 
#             networkOutputs[nbOutput] =       index
#             nbOutput += 1

#     __TrainGPU_old(
#         neuronType,
#         activationFunction,
#         alpha,
#         value,
#         biais,
#         cout,
#         delta,
#         done,
#         nbNeuronInput,
#         inputNeuronList,
#         nbNeuronOutput,
#         outputNeuronList,
#         nbInputs,
#         networkInputs,
#         nbHidden,
#         networkHidden,
#         nbOutput,
#         networkOutputs,
#         dataset,
#         expected,
#         nbEpoch,
#         learningRate
#     )
    

#     #Sauvegarde
#     for neuronName in neuronIndex.keys():       #Pour chaque neurone
#         neuronIdx = neuronIndex[neuronName]         #Index du neurone
        
#         inputData["neuronList"][neuronName]["biais"] = biais[neuronIdx]#Biais
#         for x, inputNeuron in enumerate(inputNeuronList[neuronIdx]):#Pour chaque neurone d'entrée
#             if x >= nbNeuronInput[neuronIdx]: continue               #Si on dépasse le nombre d'entrée 
#             inputData["neuronList"][neuronName]["inputList"][neuronsName[inputNeuron[0]]] = inputNeuron[1]
#         for outputNeuron in outputNeuronList[neuronIdx]:#Pour chaque neurone de sortie
#             if x >= nbNeuronOutput[neuronIdx]: continue               #Si on dépasse le nombre d'entrée 
#             inputData["neuronList"][neuronName]["outputList"][neuronsName[outputNeuron[0]]] = outputNeuron[1]


#     outputFile = open(fileName, "w")
#     json.dump(inputData, outputFile, indent = 4)
#     outputFile.close()

#     return

# @njit
# def __TrainGPU_old(
    #     neuronType: np.ndarray,         #Liste des types de neurones
    #     activationFunction: np.ndarray, #Liste des fonctions d'activation
    #     alpha: np.ndarray,              #Liste des alpha
    #     value: np.ndarray,              #Liste des valeur
    #     biais: np.ndarray,              #Liste des biais
    #     cout: np.ndarray,               #Liste des cout
    #     delta: np.ndarray,              #Liste des delta
        
    #     done: np.ndarray,               #Liste des calculé

    #     nbNeuronInput: np.ndarray,      #Pour chaque neurone, nombre de neurone d'entrée
    #     inputNeuronList: np.ndarray,    #Pour chaque neurone, liste des entrée du neurone
    #     nbNeuronOutput: np.ndarray,     #Pour chaque neurone, nombre de neurone de sortie
    #     outputNeuronList: np.ndarray,   #Pour chaque neurone, liste des sorties du neurone

    #     networkNbInput: int,            #Nombre d'entrée du réseau
    #     networkInputs: np.ndarray,      #Liste d'entrée du reseau
    #     networkNbHidden: int,           #Nombre de neurone caché du réseau
    #     networkHidden: np.ndarray,      #Liste de neurone caché du réseau
    #     networkNbOutput: int,           #Nombre de sortie du réseau
    #     networkOutputs: np.ndarray,     #Liste de neurone de sortie du réseau

    #     dataset: np.ndarray,            #Données d'entrée
    #     expected: np.ndarray,           #Sortie attendue
    #     nbEpoch: int,                   #Nombre d'entrainement
    #     learningRate: float) -> None:   #Taux d'apprentissage
    
    # leftNeuron = np.array([])
    # for epoch in range(nbEpoch):        #Pour chaque entrainement
    #     for j, inputs in enumerate(dataset):#Pour chaque données d'entrainement
    #         ####FEEDFORWARD####
    #         #RESET
    #         for x in range(len(done)):
    #             done[x] = False
    #         #INPUTS
    #         for x in range(networkNbInput):     #Pour chaque neurone d'entrée
    #             value[networkInputs[x]] = inputs[x] #On definie la valeur d'entrée
    #             done[x] = True                      #Neurone calculé
    #         #HIDDEN & OUTPUTS
    #         leftNeuron.clear()
    #         for x in networkHidden[:networkNbHidden]:
    #             leftNeuron.append(x)
    #         for x in networkOutputs[:networkNbOutput]:
    #             leftNeuron.append(x)
    #         # np.concatenate([networkHidden[:networkNbHidden],networkOutputs[:networkNbOutput]])#Liste des neurones non calculé
    #         while leftNeuron != []:         #Tant qu'il reste des neurones
    #             for x, neuronIndex in enumerate(leftNeuron):#Pour chaque neurone non calculé
    #                 #CHECK si toute les dépendances sont calculé
    #                 ready = True
    #                 for inputNeuron in inputNeuronList[neuronIndex]:#Pour chaque neurones d'entrée
    #                     if done[int(inputNeuron[0])] == False: ready = False#Si neurone d'entrées pas pret => Pas prét a calculer
    #                 if ready == False: continue         #Si neurone pas pret => neurone suivant
    #                 #CALCUL
    #                 if neuronType[neuronIndex] == NEURON_HIDDEN_CELL:#Si neurone caché
    #                     #PRE-ACTIVATION
    #                     value[neuronIndex] = 0              #Reset de la valeur pré calculé
    #                     for inputNeuron in inputNeuronList[neuronIndex]:#Pour chaque neurones d'entrée
    #                         value[neuronIndex] += inputNeuron[1] * value[int(inputNeuron[0])]#Multiplication du poids et de la valeur de l'entrée
    #                     #ACTIVATION
    #                     value[neuronIndex] = ActivationFunctionGPU(activationFunction[neuronIndex], value[neuronIndex], False, alpha[neuronIndex])#Activation du neurone
    #                 if neuronType[neuronIndex] == NEURON_OUTPUT_CELL:#Si neurone de sortie
    #                     #PRE-ACTIVATION
    #                     value[neuronIndex] = 0              #Reset de la valeur pré calculé
    #                     for inputNeuron in inputNeuronList[neuronIndex]:#Pour chaque neurones d'entrée
    #                         value[neuronIndex] += inputNeuron[1] * value[int(inputNeuron[0])]#Multiplication du poids et de la valeur de l'entrée
    #                     #ACTIVATION
    #                     value[neuronIndex] = ActivationFunctionGPU(activationFunction[neuronIndex], value[neuronIndex], False, alpha[neuronIndex])#Activation du neurone
    #                 done[neuronIndex] = True                #Neurone calculé
    #                 leftNeuron = np.delete(leftNeuron,x)    #Suppression du neurone de la liste a faire
    #                 break                                   #On recommence a 0


    #         ####BACKPROPAGATION####
    #         #RESET & INPUTS
    #         for x in range(len(done)):
    #             if neuronType[x] == NEURON_INPUT_CELL: done[x] = True
    #             else: done[x] = False
    #         #OUTPUTS
    #         for x in range(networkNbOutput):    #Pour chaque sortie
    #             neuronIndex = networkOutputs[x]
    #             cout[ neuronIndex ] = expected[j][x] - value[ neuronIndex ]#Calcul du taux d'erreur
    #             delta[ neuronIndex ] = cout[ neuronIndex ] * ActivationFunctionGPU(activationFunction[neuronIndex], value[neuronIndex], True, alpha[neuronIndex])
    #             done[neuronIndex] = True            #Neurone calculé
    #         #HIDDEN
    #         leftNeuron = networkHidden[:networkNbHidden]#Liste des neurones non calculé
    #         while len(leftNeuron) != 0:         #Tant qu'il reste des neurones
    #             for x, neuronIndex in enumerate(leftNeuron):#Pour chaque neurone non calculé
    #                 #CHECK si toute les dépendances sont calculé
    #                 ready = True
    #                 for outputNeuron in outputNeuronList[neuronIndex]:#Pour chaque neurones de sortie
    #                     if done[int(outputNeuron[0])] == False: ready = False#Si neurone de sortie pas pret => Pas prét a calculer
    #                 if ready == False: continue         #Si neurone pas pret => neurone suivant
                    
    #                 #CALCUL
    #                 if neuronType[neuronIndex] == NEURON_HIDDEN_CELL:#Si neurone caché
    #                     #PRE-ACTIVATION
    #                     cout[neuronIndex] = 0               #Remise a 0
    #                     for outputNeuron in outputNeuronList[neuronIndex]:#Pour chaque neurones de sortie
    #                         cout[neuronIndex] += delta[neuronIndex] * value[int(outputNeuron[0])]#Multiplication du poids et de la valeur du taux d'erreur
    #                     #ACTIVATION
    #                     delta[neuronIndex] = cout[neuronIndex] * ActivationFunctionGPU(activationFunction[neuronIndex], value[neuronIndex], True, alpha[neuronIndex])#Activation du neurone

    #                 done[neuronIndex] = True                #Neurone calculé
    #                 leftNeuron = np.delete(leftNeuron,x)    #Suppression du neurone de la liste a faire
    #                 break                                   #On recommence a 0


    #         ####CORRECTION####
    #         #RESET & INPUTS
    #         for x in range(len(done)):
    #             if neuronType[x] == NEURON_INPUT_CELL: done[x] = True
    #             else: done[x] = False
    #         #HIDDEN & OUTPUT
    #         leftNeuron = []
    #         for x in networkHidden[:networkNbHidden]:
    #             leftNeuron.append(x)
    #         for x in networkOutputs[:networkNbOutput]:
    #             leftNeuron.append(x)
    #         # leftNeuron = np.concatenate([networkHidden[:networkNbHidden],networkOutputs[:networkNbOutput]])#Liste des neurones non calculé
    #         while leftNeuron != []:         #Tant qu'il reste des neurones
    #             for x, neuronIndex in enumerate(leftNeuron):#Pour chaque neurone non calculé
    #                 #CHECK si toute les dépendances sont calculé
    #                 ready = True
    #                 for inputNeuron in inputNeuronList[neuronIndex]:#Pour chaque neurones d'entrée
    #                     if done[int(inputNeuron[0])] == False: ready = False#Si neurone d'entrées pas pret => Pas prét a calculer
    #                 if ready == False: continue         #Si neurone pas pret => neurone suivant
    #                 #CALCUL
    #                 if neuronType[neuronIndex] == NEURON_HIDDEN_CELL:#Si neurone caché
    #                     value[neuronIndex] = 0              #Reset de la valeur pré calculé
    #                     for inputNeuron in inputNeuronList[neuronIndex]:#Pour chaque neurones d'entrée
    #                         newWeight = inputNeuron[1] + (learningRate * delta[neuronIndex] * value[int(inputNeuron[0])])#Calcul du nouveau poids
    #                         inputNeuronList[neuronIndex][1] = newWeight #Application de la correction
    #                         outputNeuronList[ int(inputNeuron[0]) ][1] = newWeight
    #                     biais[neuronIndex] += learningRate * delta[neuronIndex]#Correction du biais
    #                 if neuronType[neuronIndex] == NEURON_OUTPUT_CELL:#Si neurone de sortie
    #                     value[neuronIndex] = 0              #Reset de la valeur pré calculé
    #                     for inputNeuron in inputNeuronList[neuronIndex]:#Pour chaque neurones d'entrée
    #                         newWeight = inputNeuron[1] + (learningRate * delta[neuronIndex] * value[int(inputNeuron[0])])#Calcul du nouveau poids
    #                         inputNeuronList[neuronIndex][1] = newWeight #Application de la correction
    #                         outputNeuronList[ int(inputNeuron[0]) ][1] = newWeight
    #                     biais[neuronIndex] += learningRate * delta[neuronIndex]#Correction du biais
    #                 done[neuronIndex] = True                #Neurone calculé
    #                 leftNeuron = np.delete(leftNeuron,x)    #Suppression du neurone de la liste a faire
    #                 break                                   #On recommence a 0
    # return