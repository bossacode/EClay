import torch
import torch.nn as nn
from dtm import DTMLayer
from pllay import PL_TopoLayer
from eclay import EC_TopoLayer





class Pllay(nn.Module):
    def __init__(self, num_classes,
                 start=0, end=7, T=32, K_max=2, dimensions=[0, 1], num_channels=1, hidden_features=[32],   # PL parameters
                 use_dtm=True, **kwargs):  # DTM parameters
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
        self.topo_layer = PL_TopoLayer(superlevel, start, end, T, K_max, dimensions, num_channels, hidden_features)
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


class Pllay2(nn.Module):
    def __init__(self, num_classes,
                 start=0, end=7, T=32, K_max=2, dimensions=[0, 1], num_channels=1, hidden_features=[32],   # PL parameters
                 start_2=1, end_2=8, K_max_2=3,                             # PL parameters 2
                 use_dtm=True, m0_1=0.05, m0_2=0.2, **kwargs):  # DTM parameters
        """
        Args:
            out_features: output dimension of fc layer
            num_classes: number of classes for classification
            use_dtm: whether to use DTM filtration
            kwargs: parameters for dtm
                    ex) m0=0.05, lims=[[1,28], [1,28]], size=[28, 28], r=2
        """
        super().__init__()
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_1 = PL_TopoLayer(False, start, end, T, K_max, dimensions, num_channels, hidden_features)
        self.topo_layer_2 = PL_TopoLayer(False, start_2, end_2, T, K_max_2, dimensions, num_channels, hidden_features)
        self.fc = nn.Linear(2*hidden_features[-1], num_classes)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]

        Returns:
            output: Tensor of shape [batch_size, num_classes]
        """
        x_1 = self.dtm_1(input)
        x_1 = self.topo_layer_1(x_1)

        x_2 = self.dtm_2(input)
        x_2 = self.topo_layer_2(x_2)

        x = torch.concat((x_1, x_2), dim=-1)
        output = self.fc(x)
        return output


class EClay(nn.Module):
    def __init__(self, num_classes,
                 start=0, end=7, T=32, num_channels=1, hidden_features=[64, 32],   # EC parameters
                 use_dtm=True, **kwargs):  # DTM parameters
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
        self.topo_layer = EC_TopoLayer(start, end, superlevel, T, num_channels, hidden_features)
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


class EClay2(nn.Module):
    def __init__(self, num_classes,
                 start=0, end=7, T=32, num_channels=1, hidden_features=[64, 32],   # EC parameters
                 start_2=1, end_2=8,                            # EC parameters 2
                 use_dtm=True, m0_1=0.05, m0_2=0.2, **kwargs):  # DTM parameters
        """
        Args:
            out_features: output dimension of fc layer
            num_classes: number of classes for classification
            use_dtm: whether to use DTM filtration
            kwargs: parameters for dtm
                    ex) m0=0.05, lims=[[1,28], [1,28]], size=[28, 28], r=2
        """
        super().__init__()
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)        ##################################have to change range
        self.topo_layer_1 = EC_TopoLayer(False, start, end, T, num_channels, hidden_features)
        self.topo_layer_2 = EC_TopoLayer(False, start_2, end_2, T, num_channels, hidden_features)
        self.fc = nn.Linear(2*hidden_features[-1], num_classes)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]

        Returns:
            output: Tensor of shape [batch_size, num_classes]
        """
        x_1 = self.dtm_1(input)
        x_1 = self.topo_layer_1(x_1)

        x_2 = self.dtm_2(input)
        x_2 = self.topo_layer_2(x_2)

        x = torch.concat((x_1, x_2), dim=-1)
        output = self.fc(x)
        return output