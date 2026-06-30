import math

from libraries import *

class NNLB: # NeuralNetworkLayoutBuilder
    def __init__(self):
        pass

    def InputLayer(self, length): # Creates a layer of Neurons
        pass

    def Hiddenlayer(self, length): # Creates a layer of Neurons
        pass

    def OutputLayer(self, length): # Creates a layer of Neurons
        pass

#   A bunch of different Activation layers

    def ReLU(self, x): # If x is greater than 0, return x. Else, return 0
        if x > 0:
            return x
        else:
            return 0

    def LeakyReLU(self, x, gradient: float = 0.01): # If x is greater than 0, return x. Else return x times the gradients
        if x >= 0:
            return x
        else:
            return x*gradient

    def PReLU(self, x, alpha: float = 0):
        if x >= 0:
            return x
        else:
            return x * alpha

    def ELU(self, x, hyperperameter: float = 1):
        if x >= 0:
            return x
        else:
            return hyperperameter * (math.exp(x) - 1)



if __name__ == "__main__":
    NNLB = NNLB()
