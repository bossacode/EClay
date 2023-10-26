import torch.nn as nn
import numpy as np
from pllay import TopoWeightLayer
from pllay_adap import AdaptiveTopoWeightLayer


class BasePllay(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.topo_layer = TopoWeightLayer(32, tseq=np.linspace(0.06, 0.3, 50), K_max=1, m0=0.05)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, 10)

    def forward(self, input):
        x = self.flatten(input)
        x = self.relu(self.topo_layer(x))
        output = self.fc(x)
        return output


class BaseAdPllay(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # self.topo_layer = AdaptiveTopoWeightLayer(32, T=50, m0=0.05, K_max=1, add=True)
        self.topo_layer = AdaptiveTopoWeightLayer(32, T=50, m0=0.05, K_max=1, add=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, 10)

    def forward(self, input):
        x = self.flatten(input)
        x = self.relu(self.topo_layer(x))
        output = self.fc(x)
        return output