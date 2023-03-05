import math
from numba import jit

#####FUNCTION#####
FUNCTION_BINARY_STEP                = 1
FUNCTION_LINEAR                     = 2
FUNCTION_SIGMOID                    = 3
FUNCTION_TANH                       = 4
FUNCTION_RELU                       = 5
FUNCTION_LEAKY_RELU                 = 6
FUNCTION_PARAMETERISED_RELU         = 7
FUNCTION_EXPONENTIAL_LINEAR_UNIT    = 8
##################

def ActivationFunction(functionType: int, z: float, prime: bool = False, alpha: float = 1) -> float:
    """
        type : "FUNCTION_#####"
        z : Pre-activation
        prime : True/False
        alpha : Default(1)
    Funtion :
        FUNCTION_BINARY_STEP (z)
        FUNCTION_LINEAR (z, alpha)
        FUNCTION_SIGMOID (z)
        FUNCTION_TANH (z)
        FUNCTION_RELU (z)
        FUNCTION_LEAKY_RELU (z, alpha)
        FUNCTION_PARAMETERISED_RELU (z, alpha)
        FUNCTION_EXPONENTIAL_LINEAR_UNIT (z, alpha)
    """
    y = 0
    if functionType == FUNCTION_BINARY_STEP:
        if not prime:
            if z < 0: y = 0
            else: y = 1
        else: 
            # pas de deriver
            pass
    if functionType == FUNCTION_LINEAR:
        if not prime:       y = z*alpha
        else:               y = alpha
    if functionType == FUNCTION_SIGMOID:
        if not prime:       y = 1/(1+math.exp(-z))
        else:               y = (1/(1+math.exp(-z))) * (1-(1/(1+math.exp(-z))))
    if functionType == FUNCTION_TANH:
        if not prime:       y = (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))
        else:               y = 1 - (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))**2
    if functionType == FUNCTION_RELU:
        if not prime:       y = max(0,z)
        else:
            if z >= 0:          y = 1
            else:               y = 0
    if functionType == FUNCTION_LEAKY_RELU:
        if not prime:       y = max(alpha*z, z)
        else:
            if z > 0:           y = 1
            else:               y = alpha
    if functionType == FUNCTION_PARAMETERISED_RELU:
        if not prime:
            if z >= 0:      y = z
            else:           y = alpha*z
        else:
            if z >= 0:          y = 1
            else:               y = alpha
    if functionType == FUNCTION_EXPONENTIAL_LINEAR_UNIT:
        if not prime:
            if z >= 0:      y = z
            else:           y = alpha*(math.exp(z)-1)
        else:
            if z >= 0:          y = z
            else:               y = alpha*(math.exp(y))
    return y

@jit(nopython=True)
def ActivationFunctionGPU(functionType: int, z: float, prime: bool = False, alpha: float = 1) -> float:
    """
        type : "FUNCTION_#####"
        z : Pre-activation
        prime : True/False
        alpha : Default(1)
    Funtion :
        FUNCTION_BINARY_STEP (z)
        FUNCTION_LINEAR (z, alpha)
        FUNCTION_SIGMOID (z)
        FUNCTION_TANH (z)
        FUNCTION_RELU (z)
        FUNCTION_LEAKY_RELU (z, alpha)
        FUNCTION_PARAMETERISED_RELU (z, alpha)
        FUNCTION_EXPONENTIAL_LINEAR_UNIT (z, alpha)
    """
    y = 0
    if functionType == FUNCTION_BINARY_STEP:
        if not prime:
            if z < 0: y = 0
            else: y = 1
        else: 
            # pas de deriver
            pass
    if functionType == FUNCTION_LINEAR:
        if not prime:       y = z*alpha
        else:               y = alpha
    if functionType == FUNCTION_SIGMOID:
        if not prime:       y = 1/(1+math.exp(-z))
        else:               y = (1/(1+math.exp(-z))) * (1-(1/(1+math.exp(-z))))
    if functionType == FUNCTION_TANH:
        if not prime:       y = (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))
        else:               y = 1 - (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))**2
    if functionType == FUNCTION_RELU:
        if not prime:       y = max(0,z)
        else:
            if z >= 0:          y = 1
            else:               y = 0
    if functionType == FUNCTION_LEAKY_RELU:
        if not prime:       y = max(alpha*z, z)
        else:
            if z > 0:           y = 1
            else:               y = alpha
    if functionType == FUNCTION_PARAMETERISED_RELU:
        if not prime:
            if z >= 0:      y = z
            else:           y = alpha*z
        else:
            if z >= 0:          y = 1
            else:               y = alpha
    if functionType == FUNCTION_EXPONENTIAL_LINEAR_UNIT:
        if not prime:
            if z >= 0:      y = z
            else:           y = alpha*(math.exp(z)-1)
        else:
            if z >= 0:          y = z
            else:               y = alpha*(math.exp(y))
    return y