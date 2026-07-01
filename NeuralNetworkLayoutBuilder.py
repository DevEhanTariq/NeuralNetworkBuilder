import math
import random

from libraries import *

class NNLB: # NeuralNetworkLayoutBuilder
    """Lets you create a Neural Network Layout"""

    def __init__(self):
        self.LeakyReLUgradient = 0.01
        self.ELUhyperperameter = 1.6732632423543772
        self.SELUhyperperameter = 1.6732632423543772
        self.BinaryTemperature = 0.5

#   A bunch of different Activation layers

    def ReLU(self, m: list): # If x is greater than 0, return x. Else, return 0
        y = []
        for x in m:
            if x > 0:
                y.append(x)
            else:
                y.append(0.0)
        return y

    def LeakyReLU(self, m: list): # If x is greater than 0, return x. Else return x times the gradients
        y = []
        for x in m:
            if x > 0:
                y.append(x)
            else:
                y.append(x*self.gradient)
        return y

    def PReLU(self, m: list, alpha: float = 0): # Learns a gradient during back probigation
        y = []
        for x, alphaVal in zip(m, alpha):
            if x > 0:
                y.append(x)
            else:
                y.append(x * alphaVal)
        return y

    def ELU(self, m: list): # A smooth gradient under 0#
        y = []
        for x in m:
            if x > 0:
                y.append(x)
            else:
                y.append(self.ELUhyperperameter * (math.exp(x) - 1))
        return y

    def SELU(self, m: list):
        lambda_ = 1.0507009873554805
        y = []
        for x in m:
            if x > 0:
                y.append(x*lambda_)
            else:
                y.append(lambda_*self.SELUhyperperameter * (math.exp(x) - 1))
        return y

    def GELU(self, m: list): # Used in Transformers and LLMs
        y = []
        for x in m:
            y.append(0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) *(x + 0.044715 * x ** 3))))
        return y

    def Swish(self, m: list):
        y = []
        for x in m:
            sigmoid = 1 / (1 + math.exp(-x))
            y.append(x * sigmoid)
        return y

    def Mish(self, m: list):
        y = []
        for x in m:
            softplus = math.log1p(math.exp(x))  # ln(1 + e^x)
            y.append(x * math.tanh(softplus))
        return y

    def Sigmoid(self, m: list):
        y = []
        for x in m:
            y.append(1 / (1 + math.exp(-x)))
        return y

    def Tanh(self, m: list):
        y = []
        for x in m:
            y.append(math.tanh(x))
        return y

    def SoftMax(self, m: list):
        expValues = []
        for x in m:
            expValues.append(math.exp(x))
        total = sum(expValues)

        y = []
        for x in expValues:
            y.append(x / total)
        return y

    def Softplus(self, m: list):
        y = []
        for x in m:
            y.append(math.log1p(math.exp(x)))
        return y

    def Identity(self, m: list):
        return m

#   Layer creators

    def layer(self, length: int, minVal: int = -1, maxVal: int = 1, doRandom: bool = True):
        if doRandom:
            return [random.uniform(minVal, maxVal) for i in range(length)]
        else:
            return [0.0 for i in range(length)]

    def IOLayer(self, length: int):
        return [None for i in range(length)]

    def Binary(self, m: list):
        expValues = []
        for x in m:
            expValues.append(math.exp(x))
        total = sum(expValues)

        softmax = []
        for x in expValues:
            softmax.append(x / total)

        y = []
        for x in softmax:
            if x >= self.BinaryTemperature:
                y.append(1)
            else:
                y.append(0)
        return y


if __name__ == "__main__":
    NNLB = NNLB()

