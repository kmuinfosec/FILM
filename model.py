import torch.nn as nn

class Invincea(nn.Module):
    def __init__(self):
        super(Invincea, self).__init__()
        self.hidden_layer_1 = nn.Linear(1025, 1025) # Input -> Hidden layer
        self.relu_1 = nn.ReLU() # Hidden Layer -> Relu
        self.hidden_layer_2 = nn.Linear(1025, 1025)
        self.relu_2 = nn.ReLU()
        self.output_layer = nn.Linear(1025, 1)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.relu_1(x)
        x = self.hidden_layer_2(x)
        x = self.relu_2(x)
        x = self.output_layer(x)
        return x

