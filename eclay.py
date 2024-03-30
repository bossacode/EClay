import torch
import torch.nn as nn
from torch_topological.nn import CubicalComplex
# import gudhi
# import numpy as np


class EC(nn.Module):
    def __init__(self, superlevel=False, tseq=[0, 7, 32], in_channels=1):
        """_summary_

        Args:
            superlevel (bool, optional): Whether to calculate topological features based on superlevel sets. If set to False, uses sublevels sets. Defaults to False.. Defaults to False.
            tseq (list, optional): List containing [start, end, number of dicretized points]. Defaults to [0, 7, 32].
            in_channels (int, optional): _description_. Defaults to 1.
        """
        super().__init__()
        self.cub_cpx = CubicalComplex(superlevel, dim=2)
        self.tseq = torch.linspace(*tseq).unsqueeze(0)  # shape: [1, T]
        self.in_channels = in_channels

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
        
        ec = torch.zeros(batch_size, self.in_channels, self.tseq.shape[-1])
        pi_list = self.cub_cpx(input)   # lists nested in order of batch_size, channel and dimension
        for b in range(batch_size):
            for c in range(self.in_channels):
                pd_0 = pi_list[b][c][0].diagram
                pd_1 = pi_list[b][c][1].diagram
                betti_0 = torch.logical_and(pd_0[:, [0]] < self.tseq, pd_0[:, [1]] >= self.tseq).sum(dim=0)
                betti_1 = torch.logical_and(pd_1[:, [0]] < self.tseq, pd_1[:, [1]] >= self.tseq).sum(dim=0)
                ec[b, c, :] = betti_0 - betti_1
        return ec if input_device == "cpu" else ec.to(input_device)


class ECLay(nn.Module):
    def __init__(self, superlevel=False, tseq=[0, 7, 32], in_channels=1, hidden_features=[64, 32]):
        """_summary_

        Args:
            superlevel (bool, optional): _description_. Defaults to False.
            tseq (list, optional): _description_. Defaults to [0, 7, 32].
            in_channels (int, optional): _description_. Defaults to 1.
            hidden_features (list, optional): _description_. Defaults to [64, 32].
        """
        super().__init__()
        self.ec_layer = EC(superlevel, tseq, in_channels)
        self.flatten = nn.Flatten()
        self.gtheta_layer = self._make_gtheta_layer(in_channels * tseq[-1], hidden_features)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        ec = self.ec_layer(input)
        ec = self.flatten(ec)   # shape: [batch_size, (num_channels * T)]
        output = self.gtheta_layer(ec)
        return output
    
    @staticmethod
    def _make_gtheta_layer(in_features, hidden_features):
        """
        Args:
            in_features:
            hidden_features:
        """
        features = [in_features] + hidden_features
        num_layers = len(hidden_features)
        layer_list = []
        for i in range(num_layers):
            layer_list.append(nn.Linear(features[i], features[i+1]))
            layer_list.append(nn.ReLU())
        return nn.Sequential(*layer_list)


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