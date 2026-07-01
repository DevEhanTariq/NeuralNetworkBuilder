import csv

from libraries import *

class MJSON: # Model CSV Builder
    def __init__(self):
        self.layers: list = []
        self.weights: list = []

    def save(self, model: list):
        for item in model:
            if type(item) == type([]):
                self.layers.append(item)

        with open("ModelLayout/Layers.json5", "w") as f:
            json5.dump(self.layers, f, indent=4)

    def saveWeights(self):
        for i in range(len(self.layers)-1):
            lengthNow = len(self.layers[i])
            lengthNext = len(self.layers[i+1])
            n = []
            for j in range(lengthNext):
                m = [random.uniform(-1, 1) for i in range(lengthNow)]
                n.append(m)
            self.weights.append(n)


        with open("ModelLayout/Weights.json5", "w") as f:
            json5.dump(self.weights, f, indent=4)


