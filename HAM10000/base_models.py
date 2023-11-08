import torch
import torch.nn as nn
import numpy as np
from pllay import TopoWeightLayer
from pllay_adap import AdTopoLayer


class BasePllay_05(nn.Module):
    def __init__(self, out_features=50):
        super().__init__()
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, lims=[[224, 0], [0, 224]], K_max=2, p=0, robust=True),
                                        )
        # self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_features, 10)

    def forward(self, input):
        x = self.topo_layer_1(input)
        # x = self.bn(x)  ################################## whether to use this or not
        #################################################################
        signal = torch.abs(x.detach()).sum(dim=0) # shape: [out_features, ]
        #################################################################
        output = self.fc(self.relu(x))
        return output, signal


class BasePllay_2(nn.Module):
    def __init__(self, out_features=50):
        super().__init__()
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.2, lims=[[224, 0], [0, 224]], K_max=2, p=0, robust=True),
                                        )
        # self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_features, 10)

    def forward(self, input):
        x = self.topo_layer_2(input)
        # x = self.bn(x)  ################################## whether to use this or not
        #################################################################
        signal = torch.abs(x.detach()).sum(dim=0) # shape: [out_features, ]
        #################################################################
        output = self.fc(self.relu(x))
        return output, signal


class BasePllay_05_2(nn.Module):
    def __init__(self, out_features=50):
        super().__init__()
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, lims=[[224, 0], [0, 224]], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.2, lims=[[224, 0], [0, 224]], K_max=2, p=0, robust=True),
                                        )
        # self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2*out_features, 10)

    def forward(self, input):
        x_1 = self.topo_layer_1(input)
        x_2 = self.topo_layer_2(input)
        x = torch.concat((x_1, x_2), dim=-1)
        # x = self.bn(x)  ################################## whether to use this or not
        #################################################################
        signal = torch.abs(x.detach()).sum(dim=0) # shape: [out_features, ]
        #################################################################
        output = self.fc(self.relu(x))
        return output, signal


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