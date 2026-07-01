from PIL.ImageChops import constant
from pandas.core.computation.ops import Constant

from NeuralNetworkLayoutBuilder import *

NNLB = NNLB()

print(NNLB.__getstate__())
print(NNLB.__doc__)

Constant = {
    'LeakyReLUgradient': 0.01,
    'ELUhyperperameter': 1.6732632423543772,
    'SELUhyperperameter': 1.6732632423543772,
    'BinaryTemperature': 0.5
}

Network = [
    NNLB.IOLayer(2),
    NNLB.LeakyReLU,
    NNLB.layer(1),
    NNLB.ReLU,
    NNLB.IOLayer(1),
    NNLB.SoftMax
]

# AndGate
I = [[0,0], [0,1], [1,0], [1,1]]
O = [[0], [0], [0], [1]]