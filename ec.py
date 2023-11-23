import torch
import torch.nn as nn
import numpy as np
import gudhi


class EC_CustomGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tseq, grid_size):
        """
        Args:
            input: Tensor of shape [(C*H*W),]
            tseq:
            grid_size:
        Returns:
        """
        cub_cpx = gudhi.CubicalComplex(dimensions=grid_size, top_dimensional_cells=input)
        cub_cpx.compute_persistence(homology_coeff_field=2, min_persistence=0)
        ec = torch.zeros(len(tseq))
        for i,t in enumerate(tseq):
            betti_num = cub_cpx.persistent_betti_numbers(t,t)
            ec[i] = betti_num[0] - betti_num[1]
        return ec
    
    @staticmethod
    def backward(ctx, up_grad):
        pass

class EC_layer(nn.Module):
    def __init__(self):
        super().__init__()