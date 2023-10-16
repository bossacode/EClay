import torch
import torch.nn as nn
import numpy as np
from pllay import TopoWeightLayer, AdaptiveTopoWeightLayer


class PllayMLP(nn.Module):
    def __init__(self, out_features=32, num_classes=10):
        super().__init__()
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        TopoWeightLayer(out_features, tseq=np.linspace(0.06, 0.3, 25), m0=0.05, K_max=2))    # hyperparameter 수정
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        TopoWeightLayer(out_features, tseq=np.linspace(0.14, 0.4, 27), m0=0.2, K_max=3))     # hyperparameter 수정
        self.fc_layer = nn.Sequential(nn.Linear(784 + out_features*2, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, num_classes))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, input):
        x_1 = self.topo_layer_1(input)
        x_1 = self.relu(x_1)
        x_2 = self.topo_layer_2(input)
        x_2 = self.relu(x_2)
        x = torch.concat((self.flatten(input), x_1, x_2), dim=-1)
        output = self.fc_layer(x)
        return output


class AdaptivePllayMLP(nn.Module):
    def __init__(self, out_features=32, num_classes=10):
        super().__init__()
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        AdaptiveTopoWeightLayer(out_features, T=25, m0=0.05, K_max=2))    # hyperparameter 수정
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        AdaptiveTopoWeightLayer(out_features, T=27, m0=0.2, K_max=3))     # hyperparameter 수정
        self.fc_layer = nn.Sequential(nn.Linear(784 + out_features*2, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, num_classes))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, input):
        x_1 = self.topo_layer_1(input)
        x_1 = self.relu(x_1)
        x_2 = self.topo_layer_2(input)
        x_2 = self.relu(x_2)
        x = torch.concat((self.flatten(input), x_1, x_2), dim=-1)
        output = self.fc_layer(x)
        return output


class CNN_Pi(nn.Module):
    def __init__(self, out_features=32, kernel_size=3):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, out_features, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(out_features, 1, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Flatten())
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        TopoWeightLayer(out_features, tseq=np.linspace(0.06, 0.3, 25), m0=0.05, K_max=2),
                                        nn.ReLU())
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        TopoWeightLayer(out_features, tseq=np.linspace(0.14, 0.4, 27), m0=0.2, K_max=3),
                                        nn.ReLU())
        self.linear_layer = nn.Sequential(nn.Linear(784+out_features+out_features, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 10))

    def forward(self, input):
        x_1 = self.conv_layer(input)
        x_2 = self.topo_layer_1(input)
        x_3 = self.topo_layer_2(input)
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        out = self.linear_layer(x)
        return out


class AdaptiveCNN_Pi(nn.Module):
    def __init__(self, out_features=32, kernel_size=3):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, out_features, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(out_features, 1, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Flatten())
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        AdaptiveTopoWeightLayer(out_features, T=25, m0=0.05, K_max=2),
                                        nn.ReLU())
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        AdaptiveTopoWeightLayer(out_features, T=27, m0=0.2, K_max=3),
                                        nn.ReLU())
        self.linear_layer = nn.Sequential(nn.Linear(784+out_features+out_features, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 10))

    def forward(self, input):
        x_1 = self.conv_layer(input)
        x_2 = self.topo_layer_1(input)
        x_3 = self.topo_layer_2(input)
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        out = self.linear_layer(x)
        return out


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1,32,kernel_size=3,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(32,1,kernel_size=3,padding=1),
                                        nn.ReLU(),
                                        nn.Flatten())
        self.linear_layer = nn.Sequential(nn.Linear(784, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 10))
    
    def forward(self, input):
        x = self.conv_layer(input)
        out = self.linear_layer(x)
        return out