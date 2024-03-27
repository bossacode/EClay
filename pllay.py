import torch
import torch.nn as nn
from torch_topological.nn import CubicalComplex
from dtm import DTMLayer
# import gudhi
# import numpy as np


class PL(nn.Module):
    def __init__(self, superlevel=False, start=0, end=7, T=32, K_max=2, dimensions=[0,1], in_channels=1):
        """
        Args:
            superlevel: Whether to calculate topological features based on superlevel sets. If set to False, uses sublevels sets
            start: Min value of domain
            end: Max value of domain
            T: How many discretized points to use
            K_max: 
            dimensions: 
            num_channels: Number of channels in input
        """
        super().__init__()
        self.cub_cpx = CubicalComplex(superlevel, dim=2)
        self.T = T
        self.tseq = torch.linspace(start, end, T).unsqueeze(0)  # shape: [1, T]
        self.K_max = K_max
        self.dimensions = dimensions
        self.len_dim = len(dimensions)
        self.in_channels = in_channels

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]
        Returns:
            landscape: Tensor of shape [batch_size, num_channels, len_dim, K_max, T]
        """
        batch_size = input.shape[0]
        input_device = input.device
        if input_device.type != "cpu":
            input = input.cpu()     # bc. calculation of persistence diagram is much faster on cpu

        landscape = torch.zeros(batch_size, self.in_channels, self.len_dim, self.K_max, self.T)
        pi_list = self.cub_cpx(input)  # lists nested in order of batch_size, channel and dimension
        for b in range(batch_size):
            for c in range(self.in_channels):
                for d, dim in enumerate(self.dimensions):
                    pd = pi_list[b][c][dim].diagram     # error if "dim" is out of range
                    pl = self._pd_to_pl(pd)
                    landscape[b, c, d, :, :] = pl
        return landscape if input_device == "cpu" else landscape.to(input_device)

    def _pd_to_pl(self, pd):
        """
        Args:
            pd: persistence diagram, shape: [n, 2]
        Returns:
            pl: persistence landscapes, shape: [K_max, T]
        """
        num_ph = pd.shape[0]    # number of homology features (= n)
        if num_ph == 0:         # no homology feature
            return torch.zeros(self.K_max, self.T)
        
        birth = pd[:, [0]]  # shape: [n, 1]
        death = pd[:, [1]]  # shape: [n, 1]
        temp = torch.zeros(max(num_ph, self.K_max), self.T)
        temp[:num_ph, :] = torch.maximum(torch.minimum(self.tseq - birth, death - self.tseq), torch.tensor(0))    # shape: [n, T]
        pl = torch.sort(temp, dim=0, descending=True).values[:self.K_max, :]    # shape: [K_max, T]
        return pl


class PLLay(nn.Module):
    def __init__(self, superlevel=False, start=0, end=7, T=32, K_max=2, dimensions=[0, 1], in_channels=1, hidden_features=[32]):
        """
        Args:
            superlevel: 
            T: 
            K_max: 
            dimensions: 
            num_channels: Number of channels in input
            hidden_features: List containing the dimension of fc layers
        """
        super().__init__()
        self.pl_layer = PL(superlevel, start, end, T, K_max, dimensions, in_channels)
        self.flatten = nn.Flatten()
        self.gtheta_layer = self._make_gtheta_layer(in_channels * len(dimensions) * K_max * T, hidden_features)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        pl = self.pl_layer(input)
        pl = self.flatten(pl)   # shape: [batch_size, (num_channels * len_dim * K_max * T)]
        output = self.gtheta_layer(pl)
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


# class AdPLCustomGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, T=100, K_max=2, grid_size=[28, 28], dimensions=[0, 1], dtype='float32'):
#         """
#         Args:
#             input: Tensor of shape [batch_size, (C*H*W)]
#             tseq:
#             K_max:
#             grid_size:
#             dimensions:
#         Returns:
#             landscape: Tensor of shape [batch_size, len_dim, T, K_max]
#             gradient: Tensor of shape [batch_size, len_dim, T, K_max, (C*H*W)]
#         """
#         device = input.device
#         land_list = []
#         diff_list = []
#         ###############################################################
#         # for loop over batch (chech if parallelizable)
#         ###############################################################
#         np_input = input.detach().cpu().numpy()
#         for n_batch in range(input.shape[0]):
#             dtm_val = np_input[n_batch]
#             cub_cpx = gudhi.CubicalComplex(dimensions=grid_size, top_dimensional_cells=dtm_val)
#             ph = cub_cpx.persistence(homology_coeff_field=2, min_persistence=0)     # list of (dimension, (birth, death))
#             ph = np.array([(dim, birth, death) for dim, (birth, death) in ph if death != float('inf')], dtype=dtype)    # exclude (birth, inf) in 0-dim homology
            
#             num_dim = len(dimensions)
#             num_ph = len(ph)

#             if num_ph == 0:  # no homology features
#                 land = np.zeros((num_dim, T, K_max), dtype=dtype)
#                 land_list.append(land)
#                 diff = np.zeros((num_dim, T, K_max, len(dtm_val)), dtype=dtype)
#                 diff_list.append(diff)
#                 continue
            
#             dim_index, birth, death = ph[:, 0], ph[:, 1], ph[:, 2]  # shape: [num_ph, ]

#             location = cub_cpx.cofaces_of_persistence_pairs()    # list of 2 lists of numpy arrays
#             location_vstack = np.vstack(location[0])    # shape: [num_ph, 2]
#             birth_location = location_vstack[:, 0]      # shape: [num_ph, ]
#             death_location = location_vstack[:, 1]

#             land = np.zeros((num_dim, T, K_max), dtype=dtype)
#             land_diff_birth = np.zeros((num_dim, T, K_max, num_ph), dtype=dtype)
#             land_diff_death = np.zeros((num_dim, T, K_max, num_ph), dtype=dtype)

#             for i_dim, dim in enumerate(dimensions):
#                 d_ind = (dim_index == dim)
#                 num_d_ph = sum(d_ind)   # number of d-dimensional homology features

#                 if num_d_ph == 0:   # no "dim"-dimensional homology features
#                     continue
                
#                 d_birth = birth[d_ind].reshape(1, -1)   # shape: [1, len_dim_ph]
#                 d_death = death[d_ind].reshape(1, -1)   # shape: [1, len_dim_ph]

#                 tseq = np.linspace(0, 1, T, dtype=dtype).reshape(-1, 1)  # shape: [T, 1]

#                 # calculate persistence landscapes
#                 fab = np.zeros((T, max(num_d_ph, K_max)), dtype=dtype)
#                 fab[:, :num_d_ph] = np.maximum(np.minimum(tseq - d_birth, d_death - tseq), 0.)

#                 temp = -np.sort(-fab, axis=-1)[:, :K_max]
#                 land[i_dim] = temp
#                 land_ind = np.argsort(-fab, axis=-1)[:, :K_max]    # shape: [T, K_max]

#                 # derivative of landscape functions with regard to persistence diagram: dL/dD
#                 # (t > birth) & (t < (birth + death)/2)
#                 fab_diff_birth = np.where((tseq > d_birth) & (2*tseq < d_birth + d_death), -1., 0.)   # shape: [T, len_dim_ph]
#                 # (t < death) & (t > (birth + death)/2)
#                 fab_diff_death = np.where((tseq < d_death) & (2*tseq > d_birth + d_death), 1., 0.)

#                 land_diff_birth[i_dim, :, :, d_ind] = np.where(np.expand_dims(land_ind, 0) == np.arange(num_d_ph).reshape(-1, 1, 1),
#                                                                 np.expand_dims(fab_diff_birth.T, -1),
#                                                                 0.)
#                 land_diff_death[i_dim, :, :, d_ind] = np.where(np.expand_dims(land_ind, 0) == np.arange(num_d_ph).reshape(-1, 1, 1),
#                                                                 np.expand_dims(fab_diff_death.T, -1),
#                                                                 0.)
#             land_list.append(land)
            
#             # derivative of persistence diagram with regard to input: dD/dX
#             diag_diff_birth = np.zeros((num_ph, len(dtm_val)), dtype=dtype)
#             diag_diff_birth[np.arange(num_ph), birth_location] = 1.

#             diag_diff_death = np.zeros((num_ph, len(dtm_val)), dtype=dtype)
#             diag_diff_death[np.arange(num_ph), death_location] = 1.

#             dimension = np.hstack([np.repeat(ldim, len(location[0][ldim])) for ldim in range(len(location[0]))])
#             persistence = dtm_val[death_location] - dtm_val[birth_location]
#             order = np.lexsort((-persistence, -dimension))

#             diff = np.dot(land_diff_birth, diag_diff_birth[order, :]) + np.dot(land_diff_death, diag_diff_death[order, :])
#             diff_list.append(diff)

#         landscape = torch.from_numpy(np.stack(land_list)).to(torch.float32).to(device)
#         gradient = torch.from_numpy(np.stack(diff_list)).to(torch.float32).to(device)
#         ctx.save_for_backward(gradient)
#         return landscape, gradient

#     @staticmethod
#     def backward(ctx, up_grad, _up_grad_gradient):
#         local_grad, = ctx.saved_tensors
#         down_grad = torch.einsum('...ijk,...ijkl->...l', up_grad, local_grad)
#         return down_grad, None, None, None, None, None


# class ScaledPLLayer(nn.Module):
#     def __init__(self, T=100, K_max=2, grid_size=[28, 28], dimensions=[0, 1]):
#         """
#         Args:
#             T: 
#             K_max: 
#             grid_size: 
#             dimensions: 
#         """
#         super().__init__()
#         self.T = T
#         self.K_max = K_max
#         self.grid_size = grid_size
#         self.dimensions = dimensions

#     def forward(self, input):
#         """
#         Args:
#             input: Tensor of shape [batch_size, (C*H*W)]
#         Returns:
#             landscape: Tensor of shape [batch_size, len_dim, T, k_max]
#         """
#         land, grad = AdPLCustomGrad.apply(input, self.T, self.K_max, self.grid_size, self.dimensions)
#         return land