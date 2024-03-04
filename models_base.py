import torch
import torch.nn as nn
from dtm import DTMLayer
from pllay_adap import AdTopoLayer
from pllay_scaled import ScaledTopoLayer
from ec import EC_TopoLayer

class BasePllay(nn.Module):
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


class ScaledPllay(nn.Module):
    def __init__(self, num_classes,
                 T=50, K_max=2, dimensions=[0, 1], num_channels=1, hidden_features=[128, 64], p=0.5,    # PL parameters
                 use_dtm=False, **kwargs):  # DTM parameters
        """
        Args:
            out_features: output dimension of fc layer
            num_classes: number of classes for classification
            use_dtm: whether to use DTM filtration
            kwargs: parameters for dtm
                    ex) m0=0.05, lims=[[1,28], [1,28]], size=[28, 28], r=2
        """
        super().__init__()
        self.use_dtm = use_dtm
        if use_dtm:
            self.dtm = DTMLayer(**kwargs, scale_dtm=True)

        superlevel = False if use_dtm else True
        self.topo_layer = ScaledTopoLayer(superlevel, T, K_max, dimensions, num_channels, hidden_features, p)
        self.fc = nn.Linear(hidden_features[-1], num_classes)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]

        Returns:
            output: Tensor of shape [batch_size, num_classes]
        """
        if self.use_dtm:
            x = self.dtm(input)
        else:
            x = input
        x = self.topo_layer(x)
        # x = self.bn(x)  ################################## whether to use this or not
        output = self.fc(self.relu(x))
        return output
    

class EClay(nn.Module):
    def __init__(self, num_classes,
                 T=50, num_channels=1, hidden_features=[128, 64], p=0.5,   # EC parameters
                 use_dtm=False, **kwargs):  # DTM parameters
        """
        Args:
            out_features: output dimension of fc layer
            num_classes: number of classes for classification
            use_dtm: whether to use DTM filtration
            kwargs: parameters for dtm
                    ex) m0=0.05, lims=[[1,28], [1,28]], size=[28, 28], r=2
        """
        super().__init__()
        self.use_dtm = use_dtm
        if use_dtm:
            self.dtm = DTMLayer(**kwargs, scale_dtm=True)

        superlevel = False if use_dtm else True
        self.topo_layer = EC_TopoLayer(superlevel, T, num_channels, hidden_features, p)
        self.fc = nn.Linear(hidden_features[-1], num_classes)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]

        Returns:
            output: Tensor of shape [batch_size, num_classes]
        """
        if self.use_dtm:
            x = self.dtm(input)
        else:
            x = input
        x = self.topo_layer(x)
        output = self.fc(x)
        return output