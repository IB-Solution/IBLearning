from ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork import *
from ConvolutionalNeuralNetwork.__ActivationFunction import *
from timeit import default_timer as timer

import matplotlib.pyplot as plt

dataset = [
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
]
expected = [
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1]
]

maxEpoch = int(input("nbEpoch maximal :"))

axisEpoch = []
axisCPUTime = []
axisGPUTime = []

brain = ConvolutionalNeuralNetwork(4, 1, NEURON_OUTPUT_CELL, FUNCTION_SIGMOID)
brain.AddLayer(8, NEURON_HIDDEN_CELL, FUNCTION_SIGMOID)
brain.SaveNetwork("testNetwork.json")
for x in range(0, maxEpoch, 10):
    axisEpoch.append(x)

    #CPU
    print("Start CPU")
    brain.LoadNetwork("testNetwork.json")
    start = timer()
    brain.Train(dataset, expected, x, 0.5, False, False)
    axisCPUTime.append(timer()-start)
    print("Done CPU")
    #GPU
    print("Start GPU")
    brain.LoadNetwork("testNetwork.json")
    start = timer()
    brain.Train(dataset, expected, x, 0.25, True, False)
    axisGPUTime.append(timer()-start)
    print("Done GPU")

    print(x, "/", maxEpoch)
    
plt.plot(axisEpoch, axisCPUTime, label="CPU")
plt.plot(axisEpoch, axisGPUTime, label="GPU")
plt.xlabel('x - nbEpoch')
plt.ylabel('y - secondes')
plt.title('Perf CNN')
plt.legend()
plt.show()

# brain.SaveNetwork("testNetwork.json")
input("end")