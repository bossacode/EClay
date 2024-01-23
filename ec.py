import torch
import torch.nn as nn
import numpy as np
import gudhi
from torch_topological.nn import CubicalComplex


# class EC_CustomGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, tseq, grid_size):
#         """
#         Args:
#             input: Tensor of shape [(C*H*W),]
#             tseq:
#             grid_size:
#         Returns:
#         """
#         cub_cpx = gudhi.CubicalComplex(dimensions=grid_size, top_dimensional_cells=input)
#         cub_cpx.compute_persistence(homology_coeff_field=2, min_persistence=0)
#         ec = torch.zeros(len(tseq))
#         for i,t in enumerate(tseq):
#             betti_num = cub_cpx.persistent_betti_numbers(t,t)
#             ec[i] = betti_num[0] - betti_num[1]
#         return ec
    
#     @staticmethod
#     def backward(ctx, up_grad):
#         pass

    
class EC_Layer(nn.Module):
    def __init__(self, T=50, num_channels=3):
        super().__init__()
        self.cub_cpx = CubicalComplex(superlevel=False, dim=2)
        self.T = T
        self.tseq = torch.linspace(0, 1, T).view(1, -1) # shape: [1, T]
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
        pi_list = self.cub_cpx(input)
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
    def __init__(self, T=50, num_channels=3, out_features=100, p=0):
        """
        Args:
            T: 
            K_max: 
            dimensions: 
            num_channels: number of channels in input
            out_features: output dimension of fc layer
            p: dropout probability
        """
        super().__init__()
        self.ec_layer = EC_Layer(T, num_channels)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p)
        self.gtheta_layer = nn.Linear(num_channels * T, out_features)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        ec = self.ec_layer(input)
        ec = self.dropout(self.flatten(ec)) # shape: [batch_size, (num_channels * T)]
        output = self.gtheta_layer(ec)
        return output