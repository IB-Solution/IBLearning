from IBLearning import *
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