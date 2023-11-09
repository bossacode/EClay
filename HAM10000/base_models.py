import torch
import torch.nn as nn
import numpy as np
from pllay import TopoWeightLayer
from pllay_adap import AdTopoLayer


class BasePllay_05(nn.Module):
    def __init__(self, out_features=50):
        super().__init__()
        self.topo_layer_11 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_12 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_13 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        # self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(3*out_features, 7)

    def forward(self, input):
        x_1 = self.topo_layer_11(input[:, 0, :, :])
        x_2 = self.topo_layer_12(input[:, 1, :, :])
        x_3 = self.topo_layer_13(input[:, 2, :, :])
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        # x = self.bn(x)  ################################## whether to use this or not
        #################################################################
        signal = torch.abs(x.detach()).sum(dim=0) # shape: [out_features, ]
        #################################################################
        output = self.fc(self.relu(x))
        return output, signal


class BasePllay_2(nn.Module):
    def __init__(self, out_features=50):
        super().__init__()
        self.topo_layer_21 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.2, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_22 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.2, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_23 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.2, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        # self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(3*out_features, 7)

    def forward(self, input):
        x_1 = self.topo_layer_21(input[:, 0, :, :])
        x_2 = self.topo_layer_22(input[:, 1, :, :])
        x_3 = self.topo_layer_23(input[:, 2, :, :])
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        # x = self.bn(x)  ################################## whether to use this or not
        #################################################################
        signal = torch.abs(x.detach()).sum(dim=0) # shape: [out_features, ]
        #################################################################
        output = self.fc(self.relu(x))
        return output, signal


class BasePllay_05_2(nn.Module):
    def __init__(self, out_features=50):
        super().__init__()
        self.topo_layer_11 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_12 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_13 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_21 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.2, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_22 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.2, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        self.topo_layer_23 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.2, lims=[[223, 0], [0, 223]], size=[224, 224], K_max=2, p=0, robust=True),
                                        )
        # self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(6*out_features, 7)

    def forward(self, input):
        x_1 = self.topo_layer_11(input[:, 0, :, :])
        x_2 = self.topo_layer_12(input[:, 1, :, :])
        x_3 = self.topo_layer_13(input[:, 2, :, :])
        x_4 = self.topo_layer_21(input[:, 0, :, :])
        x_5 = self.topo_layer_22(input[:, 1, :, :])
        x_6 = self.topo_layer_23(input[:, 2, :, :])
        x = torch.concat((x_1, x_2, x_3, x_4, x_5, x_6), dim=-1)
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