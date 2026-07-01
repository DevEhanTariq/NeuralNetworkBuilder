from NeuralNetworkBuilderAndRunner import *

# The constant values used by some of the activation functions
NNLB = NNLB(
    {
        'LeakyReLUgradient': 0.01, # gradient of the line below y=0
        'ELUhyperperameter': 1.6732632423543772, # multiplier
        'SELUhyperperameter': 1.6732632423543772, # multiplier
        'BinaryTemperature': 0.5 # minimum probability for 1
    }
)

print(NNLB.__doc__)
print(NNLB.__getstate__())

# A List Containing your Models layout
Model = [
    NNLB.IOLayer(2),
    NNLB.ReLU,
    NNLB.layer(3),
    NNLB.ReLU,
    NNLB.IOLayer(3),
    NNLB.Binary
]

# AndGate
I = [[0,0], [0,1], [1,0], [1,1]] # Training Inputs
O = [[0], [0], [0], [1]] # Expected Outputs

NNLB.modelSave(Model)