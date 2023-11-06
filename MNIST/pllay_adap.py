import numpy as np
import torch
import torch.nn as nn
import gudhi
from pllay import grid_by, DTMLayer, WALandLayer

class AdPLCustomGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, T=100, K_max=2, grid_size=[28, 28], dimensions=[0, 1], robust=True, dtype='float32'):
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
        t_range_list = []
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
                t_range_list.append([0, 0] * num_dim)
                continue
            
            dim_index, birth, death = ph[:, 0], ph[:, 1], ph[:, 2]  # shape: [num_ph, ]

            location = cub_cpx.cofaces_of_persistence_pairs()    # list of 2 lists of numpy arrays
            location_vstack = np.vstack(location[0])    # shape: [num_ph, 2]
            birth_location = location_vstack[:, 0]      # shape: [num_ph, ]
            death_location = location_vstack[:, 1]

            land = np.zeros((num_dim, T, K_max), dtype=dtype)
            land_diff_birth = np.zeros((num_dim, T, K_max, num_ph), dtype=dtype)
            land_diff_death = np.zeros((num_dim, T, K_max, num_ph), dtype=dtype)

            t_min_max = []
            for i_dim, dim in enumerate(dimensions):
                d_ind = (dim_index == dim)
                num_d_ph = sum(d_ind)   # number of d-dimensional homology features

                if num_d_ph == 0:   # no d-dimensional homology features
                    t_min_max.extend([0, 0])
                    continue
                
                d_birth = birth[d_ind].reshape(1, -1)   # shape: [1, len_dim_ph]
                d_death = death[d_ind].reshape(1, -1)   # shape: [1, len_dim_ph]

                # range selection
                if robust:
                    if num_d_ph > K_max:
                        t_min, t_max = d_birth[0][:K_max].min(), d_death[0][:K_max].max()
                    else:
                        t_min, t_max = d_birth.min(), d_death.max()
                else:
                    t_min, t_max = d_birth.min(), d_death.max()

                tseq = np.linspace(t_min, t_max, T, dtype=dtype).reshape(-1, 1)  # shape: [T, 1]

                # calculate persistence landscapes
                fab = np.zeros((T, max(num_d_ph, K_max)), dtype=dtype)
                fab[:, :num_d_ph] = np.maximum(np.minimum(tseq - d_birth, d_death - tseq), 0.)

                # add t information
                # if add_t:
                #     ######################################################################### 나눈 거 gradient 반영 안 됨
                #     temp = -np.sort(-fab, axis=-1)[:, :K_max] + tseq
                #     if temp.max() == 0: # d_birth, d_death range가 너무 짧아서 전부 0일 수도 있음
                #         land[i_dim] = temp  # 사실상 신호 없는거라 tseq 안 더함
                #     else:
                #         land[i_dim] = temp + tseq
                #     # land[i_dim] = -np.sort(-fab, axis=-1)[:, :K_max] + tseq
                # else:
                #     land[i_dim] = -np.sort(-fab, axis=-1)[:, :K_max]

                temp = -np.sort(-fab, axis=-1)[:, :K_max]
                land[i_dim] = temp
                land_ind = np.argsort(-fab, axis=-1)[:, :K_max]    # shape: [T, K_max]

                # homology 있지만 너무 작아서 사실상 신호 없는거
                if temp.max() == 0:
                    t_min_max.extend([0, 0])
                else:
                    t_min_max.extend([t_min, t_max])

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
            t_range_list.append(t_min_max)
            
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
        t_range = torch.tensor(t_range_list, dtype=torch.float32, device=device)
        ctx.save_for_backward(gradient)
        return landscape, gradient, t_range

    @staticmethod
    def backward(ctx, up_grad, _up_grad_gradient, _up_grad_t_range):
        local_grad, = ctx.saved_tensors
        down_grad = torch.einsum('...ijk,...ijkl->...l', up_grad, local_grad)
        return down_grad, None, None, None, None, None, None


class AdPLLayer(nn.Module):
    def __init__(self, T=100, K_max=2, grid_size=[28, 28], dimensions=[0, 1], robust=True):
        """
        Args:
            tseq: 
            K_max: 
            grid_size: 
            dimensions: 
        """
        super().__init__()
        self.T = T
        self.K_max = K_max
        self.grid_size = grid_size
        self.dimensions = dimensions
        self.robust = robust

    def forward(self, inputs):
        """
        Args:
            input: Tensor of shape [batch_size, (C*H*W)]
        Returns:
            landscape: Tensor of shape [batch_size, len_dim, len_tseq, k_max]
        """
        land, grad, t_range = AdPLCustomGrad.apply(inputs, self.T, self.K_max, self.grid_size, self.dimensions, self.robust)
        return land, t_range

# class AdGThetaLayer(nn.Module):
#     def __init__(self, out_features, T, K_max, dimensions=[0, 1], p=0.5):
#         """
#         Args:
#             out_features: 
#             tseq: 
#             dimensions: 
#         """
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.dropout = nn.Dropout(p=p)
#         self.g_layer = nn.Linear(T*K_max*len(dimensions) + 2*len(dimensions), out_features)

#     def forward(self, input, t_range):
#         """
#         Args:
#             input: Tensor of shape [batch_size, len_dim, T]

#         Returns:
#             output: Tensor of shape [batch_size, out_features]
#         """
#         x = torch.concat((self.flatten(input), t_range), dim=-1)
#         x = self.dropout(x)
#         output = self.g_layer(x)
#         return output


class AdTopoLayer(nn.Module):
    def __init__(self, out_features, T=50, m0=0.05, lims=[[27,0], [0,27]], size=[28, 28], r=2, K_max=2, dimensions=[0, 1], p=0.5, robust=True):
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
        self.grid = grid_by(lims, size)

        self.dtm_layer = DTMLayer(self.grid, m0, r)
        self.landscape_layer = AdPLLayer(T, K_max, size, dimensions, robust=robust)
        # self.avg_layer = WALandLayer(K_max, dimensions)
        self.gtheta_layer = AdGThetaLayer(out_features, T, K_max, dimensions, p)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, (C*H*W)]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        input_grid = self.grid.expand(input.shape[0], -1, -1).to(input.device)
        dtm_val = self.dtm_layer(input=input_grid, weight=input)
        land, t_range = self.landscape_layer(dtm_val)
        # weighted_avg_land = self.avg_layer(land)
        output = self.gtheta_layer(land, t_range)
        return output