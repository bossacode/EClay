import numpy as np
import torch
import torch.nn as nn
import gudhi
from torch_topological.nn import CubicalComplex
from torch_topological.utils import SelectByDimension
from itertools import chain
from dtm import DTMLayer


class AdPLCustomGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, T=100, K_max=2, grid_size=[28, 28], dimensions=[0, 1], dtype='float32'):
        """
        Args:
            input: Tensor of shape [batch_size, (C*H*W)]
            tseq:
            K_max:
            grid_size:
            dimensions:
        Returns:
            landscape: Tensor of shape [batch_size, len_dim, T, K_max]
            gradient: Tensor of shape [batch_size, len_dim, T, K_max, (C*H*W)]
        """
        device = input.device
        land_list = []
        diff_list = []
        ###############################################################
        # for loop over batch (chech if parallelizable)
        ###############################################################
        np_input = input.detach().cpu().numpy()
        for n_batch in range(input.shape[0]):
            dtm_val = np_input[n_batch]
            cub_cpx = gudhi.CubicalComplex(dimensions=grid_size, top_dimensional_cells=dtm_val)
            ph = cub_cpx.persistence(homology_coeff_field=2, min_persistence=0)     # list of (dimension, (birth, death))
            ph = np.array([(dim, birth, death) for dim, (birth, death) in ph if death != float('inf')], dtype=dtype)    # exclude (birth, inf) in 0-dim homology
            
            num_dim = len(dimensions)
            num_ph = len(ph)

            if num_ph == 0:  # no homology features
                land = np.zeros((num_dim, T, K_max), dtype=dtype)
                land_list.append(land)
                diff = np.zeros((num_dim, T, K_max, len(dtm_val)), dtype=dtype)
                diff_list.append(diff)
                continue
            
            dim_index, birth, death = ph[:, 0], ph[:, 1], ph[:, 2]  # shape: [num_ph, ]

            location = cub_cpx.cofaces_of_persistence_pairs()    # list of 2 lists of numpy arrays
            location_vstack = np.vstack(location[0])    # shape: [num_ph, 2]
            birth_location = location_vstack[:, 0]      # shape: [num_ph, ]
            death_location = location_vstack[:, 1]

            land = np.zeros((num_dim, T, K_max), dtype=dtype)
            land_diff_birth = np.zeros((num_dim, T, K_max, num_ph), dtype=dtype)
            land_diff_death = np.zeros((num_dim, T, K_max, num_ph), dtype=dtype)

            for i_dim, dim in enumerate(dimensions):
                d_ind = (dim_index == dim)
                num_d_ph = sum(d_ind)   # number of d-dimensional homology features

                if num_d_ph == 0:   # no "dim"-dimensional homology features
                    continue
                
                d_birth = birth[d_ind].reshape(1, -1)   # shape: [1, len_dim_ph]
                d_death = death[d_ind].reshape(1, -1)   # shape: [1, len_dim_ph]

                tseq = np.linspace(0, 1, T, dtype=dtype).reshape(-1, 1)  # shape: [T, 1]

                # calculate persistence landscapes
                fab = np.zeros((T, max(num_d_ph, K_max)), dtype=dtype)
                fab[:, :num_d_ph] = np.maximum(np.minimum(tseq - d_birth, d_death - tseq), 0.)

                temp = -np.sort(-fab, axis=-1)[:, :K_max]
                land[i_dim] = temp
                land_ind = np.argsort(-fab, axis=-1)[:, :K_max]    # shape: [T, K_max]

                # derivative of landscape functions with regard to persistence diagram: dL/dD
                # (t > birth) & (t < (birth + death)/2)
                fab_diff_birth = np.where((tseq > d_birth) & (2*tseq < d_birth + d_death), -1., 0.)   # shape: [T, len_dim_ph]
                # (t < death) & (t > (birth + death)/2)
                fab_diff_death = np.where((tseq < d_death) & (2*tseq > d_birth + d_death), 1., 0.)

                land_diff_birth[i_dim, :, :, d_ind] = np.where(np.expand_dims(land_ind, 0) == np.arange(num_d_ph).reshape(-1, 1, 1),
                                                                np.expand_dims(fab_diff_birth.T, -1),
                                                                0.)
                land_diff_death[i_dim, :, :, d_ind] = np.where(np.expand_dims(land_ind, 0) == np.arange(num_d_ph).reshape(-1, 1, 1),
                                                                np.expand_dims(fab_diff_death.T, -1),
                                                                0.)
            land_list.append(land)
            
            # derivative of persistence diagram with regard to input: dD/dX
            diag_diff_birth = np.zeros((num_ph, len(dtm_val)), dtype=dtype)
            diag_diff_birth[np.arange(num_ph), birth_location] = 1.

            diag_diff_death = np.zeros((num_ph, len(dtm_val)), dtype=dtype)
            diag_diff_death[np.arange(num_ph), death_location] = 1.

            dimension = np.hstack([np.repeat(ldim, len(location[0][ldim])) for ldim in range(len(location[0]))])
            persistence = dtm_val[death_location] - dtm_val[birth_location]
            order = np.lexsort((-persistence, -dimension))

            diff = np.dot(land_diff_birth, diag_diff_birth[order, :]) + np.dot(land_diff_death, diag_diff_death[order, :])
            diff_list.append(diff)

        landscape = torch.from_numpy(np.stack(land_list)).to(torch.float32).to(device)
        gradient = torch.from_numpy(np.stack(diff_list)).to(torch.float32).to(device)
        ctx.save_for_backward(gradient)
        return landscape, gradient

    @staticmethod
    def backward(ctx, up_grad, _up_grad_gradient):
        local_grad, = ctx.saved_tensors
        down_grad = torch.einsum('...ijk,...ijkl->...l', up_grad, local_grad)
        return down_grad, None, None, None, None, None


class ScaledPLLayer(nn.Module):
    def __init__(self, T=100, K_max=2, grid_size=[28, 28], dimensions=[0, 1]):
        """
        Args:
            T: 
            K_max: 
            grid_size: 
            dimensions: 
        """
        super().__init__()
        self.T = T
        self.K_max = K_max
        self.grid_size = grid_size
        self.dimensions = dimensions

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, (C*H*W)]
        Returns:
            landscape: Tensor of shape [batch_size, len_dim, T, k_max]
        """
        land, grad = AdPLCustomGrad.apply(input, self.T, self.K_max, self.grid_size, self.dimensions)
        return land


class PL(nn.Module):
    def __init__(self, T=100, K_max=2, dimensions=[0,1]):
        """
        Args:
            T: 
            K_max: 
            dimensions: 
        """
        super().__init__()
        self.T = T
        self.K_max = K_max
        self.dimensions = dimensions
        self.len_dim = len(dimensions)
        self.cub_cpx = CubicalComplex(superlevel=False, dim=2)
        self.select_by_dim = [(dim, SelectByDimension(min_dim=dim, max_dim=dim)) for dim in dimensions]

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]
        Returns:
            landscape: Tensor of shape [batch_size, len_dim, K_max, (T*C)]
        """
        B = input.shape[0]
        C = input.shape[1]
        landscape = torch.zeros(B, self.len_dim, self.K_max, self.T*C) # shape: [batch_size, len_dim, K_max, (T*C)]

        pers_info_list = self.cub_cpx(input)  # lists nested in order of batch_size, channel and dimension
        for data in range(B):
            for i, (dim, select_dim) in enumerate(self.select_by_dim):
                pi_list = select_dim(chain(*pers_info_list[data]))  # list of persistence informations corresponding to dimension "dim" for all channels
                assert pi_list, "dimension out of bounds"           # if empty list, dimension is out of bounds
                pl = self._pi_list_to_pl(pi_list, dim)    # shape: [K_max, (T*C)]
                landscape[data, i] = pl
        return landscape

    def _pi_list_to_pl(self, pi_list, dim):
        """
        Args:
            pi_list: list consisting of persistence informations
            dim: dimension of homology feature
        Returns:
            persistence landscapes of dimension "dim" concatenated for all channels, tensor of shape [K_max, (T*C)]
        """
        pl_list = []
        for pi in pi_list:
            pd = pi.diagram[:-1] if dim == 0 else pi.diagram    # remove (birth, inf.) for dimension 0, shape: [n, 2]
            num_ph = pd.shape[0]    # number of homology features (= n)
            if num_ph == 0:         # no homology feature
                pl = torch.zeros(self.K_max, self.T)
                pl_list.append(pl)
                continue
            
            birth = pd[:, 0].view(-1, 1)                    # shape: [n, 1]
            death = pd[:, 1].view(-1, 1)                    # shape: [n, 1]
            tseq = torch.linspace(0, 1, self.T).view(1, -1) # shape: [1, T]

            temp = torch.zeros(max(num_ph, self.K_max), self.T)
            temp[:num_ph, :] = torch.maximum(torch.minimum(tseq - birth, death - tseq), torch.tensor(0))    # shape: [n, T]
            pl = torch.sort(temp, dim=0, descending=True).values[:self.K_max, :]    # shape: [K_max, T]
            pl_list.append(pl)
        return torch.concat(pl_list, dim=-1)   # shape: [K_max, (T*C)]


class AdGThetaLayer(nn.Module):
    def __init__(self, out_features, T=100, K_max=2, dimensions=[0, 1]):
        """
        Args:
            out_features: 
            tseq: 
            dimensions: 
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.g_layer = nn.Linear(T*K_max*len(dimensions), out_features)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, len_dim, T, K_max]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        x = self.flatten(input)
        output = self.g_layer(x)
        return output


class ScaledTopoLayer(nn.Module):
    def __init__(self, out_features, m0=0.05, T=50, lims=[[1,28], [1,28]], size=[28, 28], r=2, K_max=2, dimensions=[0, 1], device="cpu"):
        """
        Args:
            out_features: 
            T: 
            m0: 
            lims: 
            size: 
            r: 
            K_max: 
            dimensions: 
        """
        super().__init__()
        self.dtm_layer = DTMLayer(m0, lims, size, r, device, scale_dtm=True)
        self.landscape_layer = ScaledPLLayer(T, K_max, size, dimensions)
        self.gtheta_layer = AdGThetaLayer(out_features, T, K_max, dimensions)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        dtm_val = self.dtm_layer(input)
        land = self.landscape_layer(dtm_val)
        output = self.gtheta_layer(land)
        return output