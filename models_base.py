import torch
import torch.nn as nn
from dtm import DTMLayer
from pllay import PL_TopoLayer
from eclay import EC_TopoLayer


class Pllay(nn.Module):
    def __init__(self, num_classes,
                 T=50, K_max=2, dimensions=[0, 1], num_channels=1, hidden_features=[128, 64],   # PL parameters
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
            self.dtm = DTMLayer(**kwargs)

        superlevel = False if use_dtm else True
        self.topo_layer = PL_TopoLayer(superlevel, T, K_max, dimensions, num_channels, hidden_features)
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
    

class EClay(nn.Module):
    def __init__(self, num_classes,
                 T=50, num_channels=1, hidden_features=[128, 64],   # EC parameters
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
            self.dtm = DTMLayer(**kwargs)

        superlevel = False if use_dtm else True
        self.topo_layer = EC_TopoLayer(superlevel, T, num_channels, hidden_features)
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