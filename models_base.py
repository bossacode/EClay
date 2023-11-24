import torch
import torch.nn as nn
from pllay_adap import AdTopoLayer
from pllay_scaled import ScaledTopoLayer


class BasePllay_05(nn.Module):
    def __init__(self, in_channels, num_classes, out_features, m0=0.05, **kwargs):
        super().__init__()
        assert in_channels in (1,3)
        self.in_channels = in_channels
        self.flatten = nn.Flatten()

        self.topo_layer_11 = AdTopoLayer(out_features, **kwargs)
        if in_channels == 3:
            self.topo_layer_12 = AdTopoLayer(out_features, **kwargs)
            self.topo_layer_13 = AdTopoLayer(out_features, **kwargs)

        # self.bn = nn.BatchNorm1d(in_channels*out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channels*out_features, num_classes)

    def forward(self, input):
        if self.in_channels == 1:
            x = self.topo_layer_11(self.flatten(input))
        else:
            x_1 = self.topo_layer_11(self.flatten(input[:, [0], :, :]))
            x_2 = self.topo_layer_12(self.flatten(input[:, [1], :, :]))
            x_3 = self.topo_layer_13(self.flatten(input[:, [2], :, :]))
            x = torch.concat((x_1, x_2, x_3), dim=-1)
        # x = self.bn(x)  ################################## whether to use this or not
        output = self.fc(self.relu(x))
        return output


class BasePllay_2(nn.Module):
    def __init__(self, in_channels, num_classes, out_features, m0=0.2, **kwargs):
        super().__init__()
        assert in_channels in (1,3)
        self.in_channels = in_channels
        self.flatten = nn.Flatten()
        
        self.topo_layer_21 = AdTopoLayer(out_features, **kwargs)
        if in_channels == 3:
            self.topo_layer_22 = AdTopoLayer(out_features, **kwargs)
            self.topo_layer_23 = AdTopoLayer(out_features, **kwargs)

        # self.bn = nn.BatchNorm1d(in_channesl*out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channels*out_features, num_classes)

    def forward(self, input):
        if self.in_channels == 1:
            x = self.topo_layer_21(self.flatten(input))
        else:
            x_1 = self.topo_layer_21(self.flatten(input[:, [0], :, :]))
            x_2 = self.topo_layer_22(self.flatten(input[:, [1], :, :]))
            x_3 = self.topo_layer_23(self.flatten(input[:, [2], :, :]))
            x = torch.concat((x_1, x_2, x_3), dim=-1)
        # x = self.bn(x)  ################################## whether to use this or not
        output = self.fc(self.relu(x))
        return output
    

class ScaledPllay_05(nn.Module):
    def __init__(self, in_channels, num_classes, out_features, m0=0.05, **kwargs):
        super().__init__()
        assert in_channels in (1,3)
        self.in_channels = in_channels
        self.flatten = nn.Flatten()

        self.topo_layer_11 = ScaledTopoLayer(out_features, **kwargs)
        if in_channels == 3:
            self.topo_layer_12 = ScaledTopoLayer(out_features, **kwargs)
            self.topo_layer_13 = ScaledTopoLayer(out_features, **kwargs)

        # self.bn = nn.BatchNorm1d(in_channels*out_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channels*out_features, num_classes)

    def forward(self, input):
        if self.in_channels == 1:
            x = self.topo_layer_11(self.flatten(input))
        else:
            x_1 = self.topo_layer_11(self.flatten(input[:, [0], :, :]))
            x_2 = self.topo_layer_12(self.flatten(input[:, [1], :, :]))
            x_3 = self.topo_layer_13(self.flatten(input[:, [2], :, :]))
            x = torch.concat((x_1, x_2, x_3), dim=-1)
        # x = self.bn(x)  ################################## whether to use this or not
        output = self.fc(self.relu(x))
        return output