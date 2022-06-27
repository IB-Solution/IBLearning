import random, time
from numba import jit, njit
import numpy as np
from .__ActivationFunction import *

NEURON_CELL = 0

NEURON_INPUT_CELL = 1
# Backfed Input Cell
# Noisy Input Cell

NEURON_HIDDEN_CELL = 4
# Probablistic Hidden Cell
# Spiking Hidden Cell

NEURON_OUTPUT_CELL = 7
# Match Input Output Cell

# Recurrent Cell
# Memory Cell
# Different Memory Cell

# Kernel
# Convolution or Pool


class Neuron:
    name = None                                 #Nom du neurone actuel
    inputList = None                            #Dictionnaire des noms des neurones d'entré {"NomNeuron":poid}
    outputList = None                           #Dictionnaire des noms des neurones de sortie {"NomNeuron":poid}
    neuronType = None                           #Type du neurone NEURON_####_CELL

    activationFunction = None
    alpha = None
    z = None                                    #Valeur pré-activation
    value = None                                #Valeur
    biais = None                                #Biais
    cout = None                                 #Taux d'erreur
    delta = None                                #Importance de l'erreur

    done = None                                 #Si le neurone est calculé


    def __init__(self, neuronName: str, neuronType: int = NEURON_CELL, activationFunction: int = FUNCTION_SIGMOID, alpha: float = 1) -> None:
        """
            Creation d'un neurone
            Params:
                neuronName : Nom du neurone
                neuronType : Type de neurone NEURON_#####_CELL
                activationFunction : Type de fonction d'activation
        """
        self.name = neuronName
        self.inputList = {}
        self.outputList = {}
        self.neuronType = neuronType

        self.activationFunction = activationFunction
        self.alpha = alpha
        self.z = 0
        self.value = 0
        self.biais = 0
        self.cout = 0
        self.delta = 0

        self.done = False

        random.seed(time.time())
        return


    def CheckInputDependance(self, neuronList: dict) -> bool:
        """
            Check si tout les inputs sont calculé
            Params:
                neuronList : Liste de tout les neurones
            Return:
                Si tout les inputs sont calculé
        """
        for neuronName in self.inputList.keys():#Pour chaque neurone d'entrée
            if neuronList[neuronName].done == False:#Si le neurone n'est pas pret
                return False                            #Les dépendances ne sont pas prêtes
        return True                             #Les dépendance sont prêtes
    def CheckOutputDependance(self, neuronList: dict) -> bool:
        """
            Check si tout les outputs sont calculé
            Params:
                neuronList : Liste de tout les neurones
            Return:
                Si tout les outputs sont calculé
        """
        for neuronName in self.outputList.keys():#Pour chaque neurone de sortie
            if neuronList[neuronName].done == False:#Si le neurone n'est pas pret
                return False                            #Les dépendances ne sont pas prêtes
        return True                             #Les dépendance sont prêtes


    def FeedForward(self, neuronList: dict) -> None:
        """
            Calcul du neurone
            Params:
                neuronList : Liste des neurones du réseau
        """
        
        if self.neuronType == NEURON_HIDDEN_CELL:
            #PRE-ACTIVATION
            self.z = 0                                  #On remet a 0 la valeur pré calculé
            for neuronName in self.inputList.keys():    #Pour chaque neurone d'entrée
                self.z += neuronList[neuronName].value * self.inputList[neuronName]#On récupère la valeur multiplier par le poids de la liaison
            #ACTIVATION
            self.value = ActivationFunction(self.activationFunction, self.z, False, self.alpha)#Activation du neurone
        if self.neuronType == NEURON_OUTPUT_CELL:
            #PRE-ACTIVATION
            self.z = 0                                  #On remet a 0 la valeur pré calculé
            for neuronName in self.inputList.keys():    #Pour chaque neurone d'entrée
                self.z += neuronList[neuronName].value * self.inputList[neuronName]#On récupère la valeur multiplier par le poids de la liaison
            #ACTIVATION
            self.value = ActivationFunction(self.activationFunction, self.z, False, self.alpha)#Activation du neurone

        self.done = True                            #Neurone calculé
        return
    def BackProgation(self, neuronList: dict, expectedValue: float = None) -> None:
        """
            Calcul du taux d'erreur des neurones
            Params:
                neuronList : Liste des neurones du reseau
                expectedValue : Valeur attendue du neurone (POUR LES OUTPUTS)
        """

        if self.neuronType == NEURON_HIDDEN_CELL:
            if expectedValue != None:                   #Si le neurone actuel doit avoir une valeur attendue (POUR LES OUTPUTS)
                self.cout = expectedValue - self.value      #Calcul du taux d'erreur
            else:                                       #Sinon
                self.cout = 0                               #Remise a 0 du taux d'erreur
                for neuronName in self.outputList.keys():   #Pour chaque neurone de sortie
                    self.cout += neuronList[neuronName].delta * self.outputList[neuronName]#On ajoute l'erreur de la liaison (Importance de l'erreur du neurone suivant et le poid de la liaison)
            self.delta = self.cout * ActivationFunction(self.activationFunction, self.value, True, self.alpha)#Calcul de l'importance de l'erreur
        if self.neuronType == NEURON_OUTPUT_CELL:
            if expectedValue != None:                   #Si le neurone actuel doit avoir une valeur attendue (POUR LES OUTPUTS)
                self.cout = expectedValue - self.value      #Calcul du taux d'erreur
            else:                                       #Sinon
                self.cout = 0                               #Remise a 0 du taux d'erreur
                for neuronName in self.outputList.keys():   #Pour chaque neurone de sortie
                    self.cout += neuronList[neuronName].delta * self.outputList[neuronName]#On ajoute l'erreur de la liaison (Importance de l'erreur du neurone suivant et le poid de la liaison)
            self.delta = self.cout * ActivationFunction(self.activationFunction, self.value, True, self.alpha)#Calcul de l'importance de l'erreur

        self.done = True                            #Neurone calculé
        return
    def UpdateWeight(self, neuronList: dict, learningRate: float) -> None:
        """
            Mise a jour du poids des liaisons entre les neurones dépendant et ce neurone
            Params:
                inputs : Liste d'entré utilisé lors du FeedForward
                learningRate : Taux d'apprentissage (0 = aucun apprentissage, 1 = Trop d'apprentissage)
        """
        if self.neuronType == NEURON_HIDDEN_CELL:
            for neuronName in self.inputList.keys():    #Pour chaque neurone d'entrée
                newWeight = self.inputList[neuronName] + (learningRate * self.delta * neuronList[neuronName].value)#Correction du poid
                self.inputList[neuronName] = newWeight      #Application de la correction
                neuronList[neuronName].outputList[self.name] = newWeight#Application de la correction a l'autre bout de la liaison
            self.biais += learningRate * self.delta     #Mise a jour du biais
        if self.neuronType == NEURON_OUTPUT_CELL:
            for neuronName in self.inputList.keys():    #Pour chaque neurone d'entrée
                newWeight = self.inputList[neuronName] + (learningRate * self.delta * neuronList[neuronName].value)#Correction du poid
                self.inputList[neuronName] = newWeight      #Application de la correction
                neuronList[neuronName].outputList[self.name] = newWeight#Application de la correction a l'autre bout de la liaison
            self.biais += learningRate * self.delta     #Mise a jour du biais

        self.delta = 0
        self.done = True                            #Neurone corrigé
        return


    def FeedForwardGPU(self, neuronList: dict) -> None:
        """
            Calcul du neurone
            Params:
                neuronList : Liste des neurones du réseau
        """
        
        if self.neuronType == NEURON_HIDDEN_CELL:
            #PRE-ACTIVATION
            self.z = 0                                  #On remet a 0 la valeur pré calculé
            for neuronName in self.inputList.keys():    #Pour chaque neurone d'entrée
                self.z += neuronList[neuronName].value * self.inputList[neuronName]#On récupère la valeur multiplier par le poids de la liaison
            #ACTIVATION
            self.value = ActivationFunctionGPU(self.activationFunction, self.z, False, self.alpha)#Activation du neurone
        if self.neuronType == NEURON_OUTPUT_CELL:
            #PRE-ACTIVATION
            self.z = 0                                  #On remet a 0 la valeur pré calculé
            for neuronName in self.inputList.keys():    #Pour chaque neurone d'entrée
                self.z += neuronList[neuronName].value * self.inputList[neuronName]#On récupère la valeur multiplier par le poids de la liaison
            #ACTIVATION
            self.value = ActivationFunctionGPU(self.activationFunction, self.z, False, self.alpha)#Activation du neurone

        self.done = True                            #Neurone calculé
        return
    def BackProgationGPU(self, neuronList: dict, expectedValue: float = None) -> None:
        """
            Calcul du taux d'erreur des neurones
            Params:
                neuronList : Liste des neurones du reseau
                expectedValue : Valeur attendue du neurone (POUR LES OUTPUTS)
        """

        if self.neuronType == NEURON_HIDDEN_CELL:
            if expectedValue != None:                   #Si le neurone actuel doit avoir une valeur attendue (POUR LES OUTPUTS)
                self.cout = expectedValue - self.value      #Calcul du taux d'erreur
            else:                                       #Sinon
                self.cout = 0                               #Remise a 0 du taux d'erreur
                for neuronName in self.outputList.keys():   #Pour chaque neurone de sortie
                    self.cout += neuronList[neuronName].delta * self.outputList[neuronName]#On ajoute l'erreur de la liaison (Importance de l'erreur du neurone suivant et le poid de la liaison)
            self.delta = self.cout * ActivationFunctionGPU(self.activationFunction, self.value, True, self.alpha)#Calcul de l'importance de l'erreur
        if self.neuronType == NEURON_OUTPUT_CELL:
            if expectedValue != None:                   #Si le neurone actuel doit avoir une valeur attendue (POUR LES OUTPUTS)
                self.cout = expectedValue - self.value      #Calcul du taux d'erreur
            else:                                       #Sinon
                self.cout = 0                               #Remise a 0 du taux d'erreur
                for neuronName in self.outputList.keys():   #Pour chaque neurone de sortie
                    self.cout += neuronList[neuronName].delta * self.outputList[neuronName]#On ajoute l'erreur de la liaison (Importance de l'erreur du neurone suivant et le poid de la liaison)
            self.delta = self.cout * ActivationFunctionGPU(self.activationFunction, self.value, True, self.alpha)#Calcul de l'importance de l'erreur

        self.done = True                            #Neurone calculé
        return

    # def FeedForwardGPU(self, neuronList: dict) -> None:
    #     """
    #         Calcul du neurone
    #         Params:
    #             neuronList : Liste des neurones du réseau
    #     """

    #     valueList = List()
    #     weightList = List()
    #     for x, neuronName in enumerate(self.inputList.keys()):
    #         valueList.append(neuronList[neuronName].value)
    #         weightList.append(self.inputList[neuronName])

    #     self.value = FeedForwardGPU(
    #         self.neuronType,
    #         self.activationFunction,
    #         self.alpha,
    #         self.z,
    #         self.value,
    #         valueList,
    #         weightList
    #     )

    #     self.done = True                            #Neurone calculé
    #     return
    # def BackProgationGPU(self, neuronList: dict, expectedValue: float = None) -> None:
    #     """
    #         Calcul du taux d'erreur des neurones
    #         Params:
    #             neuronList : Liste des neurones du reseau
    #             expectedValue : Valeur attendue du neurone (POUR LES OUTPUTS)
    #     """

    #     deltaList = np.zeros(len(self.outputList), dtype=np.float64)
    #     weightList = np.zeros(len(self.outputList), dtype=np.float64)
    #     for x, neuronName in enumerate(self.outputList.keys()):
    #         deltaList[x] = neuronList[neuronName].delta
    #         weightList[x] = self.outputList[neuronName]

    #     self.delta = BackPropagationGPU(
    #         self.neuronType,
    #         self.activationFunction,
    #         self.alpha,
    #         self.cout,
    #         self.delta,
    #         self.value,
    #         deltaList,
    #         weightList,
    #         expectedValue
    #     )

    #     self.done = True                            #Neurone calculé
    #     return


    def AddInput(self, neuronName: str, weight: float = None) -> float:
        """
            Ajout d'un neurone dans la liste d'entré
            Param:
                neuronName : Nom du neurone a ajouter
                weight : Poid de l'entrée (si pas définie, aleatoire)
            Return:
                Poid du neurones
        """
        if not self.__CheckInputList(neuronName):
            if weight == None:
                self.inputList[neuronName] = random.random()
            else:
                self.inputList[neuronName] = weight
        return self.inputList[neuronName]
    def AddOutput(self, neuronName: str, weight: float = None) -> float:
        """
            Ajout d'un neurone dans la liste de sortie
            Param:
                neuronName : Nom du neurone a ajouter
                weight : Poid de sortie (si pas définie, aleatoire)
            Return:
                Poid du neurones
        """
        if not self.__CheckOutputList(neuronName):
            if weight == None:
                self.outputList[neuronName] = random.random()
            else:
                self.outputList[neuronName] = weight
        return self.outputList[neuronName]

    def RemoveInput(self, neuronName: str) -> None:
        """
            Supprime un neurone dans la liste d'entrée
            Param:
                neuronName : Nom du neurone a supprimer
        """
        if self.__CheckInputList(neuronName):
            del self.inputList[neuronName]
        return
    def RemoveOutput(self, neuronName: str) -> None:
        """
            Supprime un neurone dans la liste de sortie
            Param:
                neuronName : Nom du neurone a supprimer
        """
        if self.__CheckOutputList(neuronName):
            del self.outputList[neuronName]
        return
    

    def __CheckInputList(self, neuronName: str) -> bool:
        """
            Rechercher un neurone dans la liste d'entré
            Param:
                neuronName : Nom du neurone
            Return:
                Si le neurone est dans la liste
        """
        return neuronName in self.inputList.keys()
    def __CheckOutputList(self, neuronName: str) -> int:
        """
            Rechercher un neurone dans la liste de sortie
            Param:
                neuronName : Nom du neurone
            Return:
                Si le neurone est dans la liste
        """
        return neuronName in self.outputList.keys()



# @njit
# def FeedForwardGPU(
#         neuronType: int,
#         activationFunction: int,
#         alpha: float,
#         z: float,
#         value: float,
#         valueList: List,
#         weightList: List
#     ) -> float:
    
#     if neuronType == NEURON_HIDDEN_CELL:
#         #PRE-ACTIVATION
#         z = 0                                   #On remet a 0 la valeur pré calculé
#         for x in range(len(valueList)):         #Pour chaque neurone d'entrée
#             z += valueList[x] * weightList[x]       #On récupère la valeur multiplier par le poids de la liaison
#         #ACTIVATION
#         value = ActivationFunctionGPU(activationFunction, z, False, alpha)#Activation du neurone
#     if neuronType == NEURON_OUTPUT_CELL:
#         #PRE-ACTIVATION
#         z = 0                                   #On remet a 0 la valeur pré calculé
#         for x in range(len(valueList)):         #Pour chaque neurone d'entrée
#             z += valueList[x] * weightList[x]       #On récupère la valeur multiplier par le poids de la liaison
#         #ACTIVATION
#         value = ActivationFunctionGPU(activationFunction, z, False, alpha)#Activation du neurone

#     return value
# @njit
# def BackPropagationGPU(
#         neuronType: int,
#         activationFunction: int,
#         alpha: float,
#         cout: float,
#         delta: float,
#         value: float,
#         deltaList: np.ndarray,
#         weightList: np.ndarray,
#         expectedValue: float = None
#     ) -> float:

#     if neuronType == NEURON_HIDDEN_CELL:
#         if expectedValue != None:               #Si le neurone actuel doit avoir une valeur attendue (POUR LES OUTPUTS)
#             cout = expectedValue - value            #Calcul du taux d'erreur
#         else:                                   #Sinon
#             cout = 0                                #Remise a 0 du taux d'erreur
#             for x in range(len(deltaList)):         #Pour chaque neurone de sortie
#                 cout += deltaList[x] * weightList[x]    #On ajoute l'erreur de la liaison (Importance de l'erreur du neurone suivant et le poid de la liaison)
#         delta = cout * ActivationFunctionGPU(activationFunction, value, True, alpha)#Calcul de l'importance de l'erreur
#     if neuronType == NEURON_OUTPUT_CELL:
#         if expectedValue != None:               #Si le neurone actuel doit avoir une valeur attendue (POUR LES OUTPUTS)
#             cout = expectedValue - value            #Calcul du taux d'erreur
#         else:                                   #Sinon
#             cout = 0                                #Remise a 0 du taux d'erreur
#             for x in range(len(deltaList)):         #Pour chaque neurone de sortie
#                 cout += deltaList[x] * weightList[x]    #On ajoute l'erreur de la liaison (Importance de l'erreur du neurone suivant et le poid de la liaison)
#         delta = cout * ActivationFunctionGPU(activationFunction, value, True, alpha)#Calcul de l'importance de l'erreur

#     return delta