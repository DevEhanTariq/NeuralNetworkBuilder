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

Name = "AndGateAI" # Models Name

# A List Containing your Models layout
Model = [
    NNLB.ILayer(2),
    NNLB.ReLU,
    NNLB.layer(3, minVal=0),
    NNLB.ReLU,
    NNLB.layer(1, minVal=0),
    NNLB.Identity
]

# AndGate
I = [[0,0], [0,1], [1,0], [1,1]] # Training Inputs
O = [[0], [0], [0], [1]] # Expected Outputs

NNLB.modelSave(Model, Name)
ModelValues = NNLB.loadModelVariables(Model, Name)

for i in range(len(I)):
    NNLB.modelRun(ModelValues, I, O, index=i)