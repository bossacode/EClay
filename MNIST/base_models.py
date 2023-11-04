import torch
import torch.nn as nn
import numpy as np
from pllay import TopoWeightLayer
from pllay_adap import AdTopoLayer


# class BasePllay(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.topo_layer = nn.Sequential(nn.Flatten(),
#                                         TopoWeightLayer(32, tseq=np.linspace(0.06, 0.3, 25), m0=0.05, K_max=2),
#                                         nn.ReLU())
#         self.fc = nn.Linear(32, 10)

#     def forward(self, input):
#         x = self.topo_layer(input)
#         output = self.fc(x)
#         return output


class Pllay(nn.Module):
    def __init__(self):
        super().__init__()
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(32, T=25, m0=0.05, lims=[[27, 0], [0, 27]], K_max=2, p=0, robust=True),
                                        nn.ReLU())
        # self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        # AdaptiveTopoWeightLayer(100, T=100, m0=0.2, lims=[[27, 0], [0, 27]], K_max=2, p=0, robust=True),
                                        # nn.ReLU())
        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(100, 10)
        # self.fc = nn.Linear(200, 10)

    def forward(self, input):
        x_1 = self.topo_layer_1(input)
        # x_2 = self.topo_layer_2(input)
        # x = self.dropout(x)
        # x = torch.concat((x_1, x_2), dim=-1)
        output = self.fc(x_1)
        return output


# class BaseAdPllay_not_robust(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.topo_layer = AdaptiveTopoWeightLayer(32, T=25, m0=0.05, lims=[[27, 0], [0, 27]], K_max=2, robust=False)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(32, 10)

#     def forward(self, input):
#         x = self.flatten(input)
#         x = self.relu(self.topo_layer(x))
#         output = self.fc(x)
#         return output


# class BaseAdPllay_no_t(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.topo_layer = AdaptiveTopoWeightLayer_NR(32, T=25, m0=0.05, lims=[[27, 0], [0, 27]], K_max=2, robust=True)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(32, 10)

#     def forward(self, input):
#         x = self.flatten(input)
#         x = self.relu(self.topo_layer(x))
#         output = self.fc(x)
#         return output


# class BaseAdPllay_no_t_not_robust(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.topo_layer = AdaptiveTopoWeightLayer_NR(32, T=25, m0=0.05, lims=[[27, 0], [0, 27]], K_max=2, robust=False)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(32, 10)

#     def forward(self, input):
#         x = self.flatten(input)
#         x = self.relu(self.topo_layer(x))
#         output = self.fc(x)
#         return output