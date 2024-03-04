import torch
import torch.nn as nn
import numpy as np
import gudhi
from torch_topological.nn import CubicalComplex

    
class EC_Layer(nn.Module):
    def __init__(self, superlevel=False, T=50, num_channels=1):
        """
        Args:
            superlevel: Whether to calculate topological features based on superlevel sets. If set to False, uses sublevels sets
            T: How many discretized points to use
            num_channels: Number of channels in input
        """
        super().__init__()
        self.cub_cpx = CubicalComplex(superlevel=superlevel, dim=2)
        self.T = T
        start, end = (-1, 0) if superlevel else (0, 1)
        self.tseq = torch.linspace(start, end, T).unsqueeze(0)  # shape: [1, T]
        self.num_channels = num_channels

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]
        Returns:
            ec: Tensor of shape [batch_size, C, T]
        """
        batch_size = input.shape[0]
        input_device = input.device
        if input_device.type != "cpu":
            input = input.cpu()     # bc. calculation of persistence diagram is much faster on cpu
        
        ec = torch.zeros(batch_size, self.num_channels, self.T)
        pi_list = self.cub_cpx(input)   # lists nested in order of batch_size, channel and dimension
        for b in range(batch_size):
            for c in range(self.num_channels):
                pd_0 = pi_list[b][c][0].diagram
                pd_1 = pi_list[b][c][1].diagram
                betti_0 = torch.logical_and(pd_0[:, 0].view(-1, 1) < self.tseq, pd_0[:, 1].view(-1, 1) >= self.tseq).sum(dim=0)
                betti_1 = torch.logical_and(pd_1[:, 0].view(-1, 1) < self.tseq, pd_1[:, 1].view(-1, 1) >= self.tseq).sum(dim=0)
                ec[b, c, :] = betti_0 - betti_1
        return ec if input_device == "cpu" else ec.to(input_device)
    

# class EC_Layer2(nn.Module):
#     def __init__(self, T=50):
#         super().__init__()
#         self.tseq = torch.linspace(0, 1, T)

#     def forward(self, input, grid_size):
#         """
#         Args:
#             input: Tensor of shape [(C*H*W),]
#             tseq:
#             grid_size:
#         Returns:
#         """
#         cub_cpx = gudhi.CubicalComplex(dimensions=grid_size, top_dimensional_cells=input)
#         cub_cpx.compute_persistence(homology_coeff_field=2, min_persistence=0)
#         ec = torch.zeros(len(self.tseq))
#         for i, t in enumerate(self.tseq):
#             betti_num = cub_cpx.persistent_betti_numbers(t,t)
#             ec[i] = betti_num[0] - betti_num[1]
#         return ec
    

class EC_TopoLayer(nn.Module):
    def __init__(self, superlevel=False, T=50, num_channels=1, hidden_features=[128, 64], p=0.5):
        """
        Args:
            superlevel: 
            T: 
            num_channels: Number of channels in input
            hidden_features: List containing the dimension of fc layers
            p: Dropout probability
        """
        super().__init__()
        self.ec_layer = EC_Layer(superlevel, T, num_channels)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p)
        self.gtheta_layer = self._make_gtheta_layer(num_channels * T, hidden_features, p)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        ec = self.ec_layer(input)
        ec = self.flatten(ec)   # shape: [batch_size, (num_channels * T)]
        ec = self.dropout(ec)   # should i use this dropout? seems better to use?
        output = self.gtheta_layer(ec)
        return output
    
    @staticmethod
    def _make_gtheta_layer(in_features, hidden_features, p):
        """
        Args:
            in_features:
            hidden_features:
            p: Dropout probability
        """
        features = [in_features] + hidden_features
        num_layers = len(hidden_features)
        layer_list = []
        for i in range(num_layers):
            layer_list.append(nn.Linear(features[i], features[i+1]))
            # if i+1 != num_layers:
            # layer_list.append(nn.BatchNorm1d(features[i+1])) # should i use BN?
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p))
        return nn.Sequential(*layer_list)