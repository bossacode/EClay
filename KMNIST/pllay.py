import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import gudhi as gd
import numpy as np


class PL(Function):
    @staticmethod
    def forward(ctx, x, func, steps, K_max, dimensions):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W].
            func (function or method): Function that calculates PL and gradient if needed.
            steps (int): Number of discretized points.
            K_max (int): How many landscapes to use per dimension.
            dimensions (list): Homology dimensions to consider.

        Returns:
            torch.Tensor: Tensor of shape [B, C, len_dim, K_max, steps].
        """
        backprop = x.requires_grad
        device = x.device
        pl_collec = np.zeros((x.shape[0], x.shape[1], len(dimensions), K_max, steps))
        grad_collec = np.zeros((x.shape[0], x.shape[1], len(dimensions), K_max, steps, x.shape[2]*x.shape[3])) if backprop else None
        for b, data in enumerate(x.cpu().numpy()):  # iterate over batch
            for c, channel in enumerate(data):      # iterate over channel
                pl, grad = func(channel, backprop)
                pl_collec[b, c] = pl
                if backprop:
                    grad_collec[b, c] = grad
        if backprop:
            ctx.save_for_backward(torch.from_numpy(grad_collec).to(device).to(torch.float32))
            ctx.input_size = x.shape
        return torch.from_numpy(pl_collec).to(device).to(torch.float32)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """_summary_

        Args:
            grad_output (torch.Tensor): Gradient w.r.t. to output. Tensor of shape [B, C, len_dim, K_max, steps].

        Returns:
            _type_: _description_
        """
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_local, = ctx.saved_tensors                                                 # shape: [B, C, len_dim, K_max, steps, H*W]
            grad_input = torch.einsum('...ijk,...ijkl->...l', [grad_output, grad_local])    # shape: [B, C, H*W]
            grad_input = grad_input.view(*ctx.input_size)                                   # shape: [B, C, H, W]
        return grad_input, None, None, None, None


class CubPL2d(nn.Module):
    def __init__(self, as_vertices=True, sublevel=True, interval=[0.03, 0.34], steps=32, K_max=2, dimensions=[0, 1]):
        """_summary_

        Args:
            as_vertices (bool, optional): Use V-construction. If False, T-construction will be used. Defaults to True.
            sublevel (bool, optional): Use sublevel set filtration. If False, superlevel set filtration will be used by imposing a sublevel set filtration on negative data. Defaults to True.
            interval (list, optional): Minimum and maximum value of interval to consider. Defaults to [0.02, 0.28].
            steps (int, optional): Number of discretized points. Defaults to 32.
            K_max (int, optional): How many landscapes to use per dimension. Defaults to 2.
            dimensions (list, optional): Homology dimensions to consider. Defaults to [0, 1].
        """
        super().__init__()
        self.as_vertices = as_vertices
        self.sublevel = sublevel
        self.t_min, self.t_max = interval
        self.steps = steps
        self.tseq = np.linspace(*interval, steps)
        self.K_max = K_max
        self.dimensions = dimensions
    
    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Tensor of shape [B, C, len_dim, K_max, steps].
        """
        return PL.apply(x, self._cal_pl, self.steps, self.K_max, self.dimensions)

    def _cal_pl(self, x, backprop):
        """_summary_

        Args:
            x (numpy.ndarray): Numpy array of shape [H, W].
            backprop (bool): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        if self.as_vertices:
            cub_cpx = gd.CubicalComplex(vertices=x.T if self.sublevel else -x.T)    # transpose data bc. gudhi uses column-major order
            pd = cub_cpx.persistence(homology_coeff_field=2)                        # persistence diagram, list of (dimension, (birth, death))
            location = cub_cpx.vertices_of_persistence_pairs()                      # pixel indexes corresponding to birth and death, list containing 2 lists of numpy arrays
        else:
            cub_cpx = gd.CubicalComplex(top_dimensional_cells=x.T if self.sublevel else -x.T)
            pd = cub_cpx.persistence(homology_coeff_field=2)
            location = cub_cpx.cofaces_of_persistence_pairs()

        if location[0]:                                                         # homology feature exists other than 0-dim homology that persists forever
            location_vstack = [np.vstack(location[0]), np.vstack(location[1])]  # first element is birth and death locations for homology features, second element is birth location of 0-dim homology that persists forever
        else:
            location_vstack = [np.zeros((0, 2), dtype=np.int32), np.vstack(location[1])]
        birth_location = np.concatenate((location_vstack[0][:, 0], location_vstack[1][:, 0])).astype(np.int32)
        death_location = location_vstack[0][:, 1].astype(np.int32)

        len_dim = len(self.dimensions)
        len_pd = len(pd)

        pl = np.zeros((len_dim, self.K_max, self.steps))    # shape: [len_dim, K_max, steps]
        if backprop:
            pl_diff_b = np.zeros((len_dim, self.K_max, self.steps, len_pd))
            pl_diff_d = np.zeros((len_dim, self.K_max, self.steps, len_pd))

        for i_dim, dim in enumerate(self.dimensions):
            pd_dim = [pair for pair in pd if pair[0] == dim]
            pd_dim_ids = np.array([i for i, pair in enumerate(pd) if pair[0] == dim])
            len_pd_dim = len(pd_dim)    # number of "dim"-dimensional homology features

            # calculate persistence landscape
            land = np.zeros((max(len_pd_dim, self.K_max), self.steps))
            for d in range(len_pd_dim):
                for t in range(self.steps):
                    land[d, t] = max(min(self.tseq[t] - pd_dim[d][1][0], pd_dim[d][1][1] - self.tseq[t]), 0)
            pl[i_dim] = -np.sort(-land, axis=0)[:self.K_max]
            
            # calculation of gradient only for inputs that require gradient
            if backprop:
                # derivative of landscape functions with regard to persistence diagram: dPL/dPD
                land_idx = np.argsort(-land, axis=0)[:self.K_max]
                # (t > birth) & (t < (birth + death)/2)
                land_diff_b = np.zeros((len_pd_dim, self.steps))
                for d in range(len_pd_dim):
                    land_diff_b[d, :] = np.where((self.tseq > pd_dim[d][1][0]) & (2 * self.tseq < pd_dim[d][1][0] + pd_dim[d][1][1]), -1., 0.)
                # (t < death) & (t > (birth + death)/2)
                land_diff_d = np.zeros((len_pd_dim, self.steps))
                for d in range(len_pd_dim):
                    land_diff_d[d, :] = np.where((self.tseq < pd_dim[d][1][1]) & (2 * self.tseq > pd_dim[d][1][0] + pd_dim[d][1][1]), 1., 0.)

                for d in range(len_pd_dim):
                    pl_diff_b[i_dim, :, :, pd_dim_ids[d]] = np.where(d == land_idx, np.repeat(np.expand_dims(land_diff_b[d, :], axis=0), self.K_max, axis=0), 0)
                for d in range(len_pd_dim):
                    pl_diff_d[i_dim, :, :, pd_dim_ids[d]] = np.where(d == land_idx, np.repeat(np.expand_dims(land_diff_d[d, :], axis=0), self.K_max, axis=0), 0)
        
        # calculation of gradient only for inputs that require gradient
        if backprop:
            # derivative of persistence diagram with regard to input: dPD/dX
            pd_diff_b = np.zeros((len_pd, x.shape[0]*x.shape[1]))
            for i in range(len(birth_location)):
                pd_diff_b[i, birth_location[i]] = 1
            pd_diff_d = np.zeros((len_pd, x.shape[0]*x.shape[1]))
            for i in range(len(death_location)):
                pd_diff_d[i, death_location[i]] = 1	

            if location[0]:
                dimension = np.concatenate((np.hstack([np.repeat(ldim, len(location[0][ldim])) for ldim in range(len(location[0]))]),
                                            np.hstack([np.repeat(ldim, len(location[1][ldim])) for ldim in range(len(location[1]))])))
            else:
                dimension = np.hstack([np.repeat(ldim, len(location[1][ldim])) for ldim in range(len(location[1]))])
            if len(death_location) > 0:
                persistence = np.concatenate((x.reshape(-1)[death_location], np.repeat(np.infty, len(np.vstack(location[1]))))) - x.reshape(-1)[birth_location]
            else:
                persistence = np.repeat(np.infty, len(np.vstack(location[1])))
            order = np.lexsort((-persistence, -dimension))
            grad_local = np.matmul(pl_diff_b, pd_diff_b[order, :]) + np.matmul(pl_diff_d, pd_diff_d[order, :])  # shape: [len_dim, K_max, steps, H*W)
        else:
            grad_local = None
        return pl, grad_local


class GThetaPL(nn.Module):
    def __init__(self, num_features=[128, 32]):
        """
        Args:
            num_features (list, optional): List containing the size of each layer. Defaults to [128, 32].
        """
        super().__init__()
        self.gtheta = self._make_gtheta_layer(num_features)

    def forward(self, x):
        return self.gtheta(x)

    @staticmethod
    def _make_gtheta_layer(features):
        """
        Args:
            in_features:
            hidden_features:
        """
        num_layers = len(features) - 1
        layer_list = []
        for i in range(num_layers):
            layer_list.append(nn.Linear(features[i], features[i+1]))
            layer_list.append(nn.ReLU())
        return nn.Sequential(*layer_list)


class PLLay(nn.Module):
    def __init__(self, as_vertices=True, sublevel=True, interval=[0.03, 0.34], steps=32, K_max=2, dimensions=[0, 1], in_channels=1, hidden_features=[32]):
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
        self.pl_layer = CubPL2d(as_vertices, sublevel, interval, steps, K_max, dimensions)
        self.flatten = nn.Flatten()
        self.gtheta = GThetaPL([in_channels * len(dimensions) * K_max * steps] + hidden_features)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        pl = self.flatten(self.pl_layer(input)) # shape: [batch_size, (C * len_dim * K_max * steps)]
        return self.gtheta(pl)


# class CubPL2d(nn.Module):
#     def __init__(self, superlevel=False, tseq=[0, 7, 32], K_max=2, dimensions=[0,1], in_channels=1):
#         """_summary_

#         Args:
#             superlevel (bool, optional): _description_. Defaults to False.
#             tseq (list, optional): _description_. Defaults to [0, 7, 32].
#             K_max (int, optional): _description_. Defaults to 2.
#             dimensions (list, optional): _description_. Defaults to [0,1].
#             in_channels (int, optional): _description_. Defaults to 1.
#         """
#         super().__init__()
#         self.cub_cpx = CubicalComplex(superlevel, dim=2)
#         self.tseq = torch.linspace(*tseq).unsqueeze(0)  # shape: [1, T]
#         self.K_max = K_max
#         self.dimensions = dimensions
#         self.len_dim = len(dimensions)
#         self.in_channels = in_channels

#     def forward(self, input):
#         """
#         Args:
#             input: Tensor of shape [batch_size, num_channels, H, W]
#         Returns:
#             landscape: Tensor of shape [batch_size, num_channels, len_dim, K_max, T]
#         """
#         batch_size = input.shape[0]
#         input_device = input.device
#         if input_device.type != "cpu":
#             input = input.cpu()     # bc. calculation of persistence diagram is much faster on cpu

#         landscape = torch.zeros(batch_size, self.in_channels, self.len_dim, self.K_max, self.tseq.shape[-1])
#         pi_list = self.cub_cpx(input)  # lists nested in order of batch_size, channel and dimension
#         for b in range(batch_size):
#             for c in range(self.in_channels):
#                 for d, dim in enumerate(self.dimensions):
#                     pd = pi_list[b][c][dim].diagram     # error if "dim" is out of range
#                     pl = self._pd_to_pl(pd)
#                     landscape[b, c, d, :, :] = pl
#         return landscape if input_device == "cpu" else landscape.to(input_device)

#     def _pd_to_pl(self, pd):
#         """
#         Args:
#             pd: persistence diagram, shape: [n, 2]
#         Returns:
#             pl: persistence landscapes, shape: [K_max, T]
#         """
#         num_ph = pd.shape[0]    # number of homology features (= n)
#         if num_ph == 0:         # no homology feature
#             return torch.zeros(self.K_max, len(self.tseq))
        
#         birth = pd[:, [0]]  # shape: [n, 1]
#         death = pd[:, [1]]  # shape: [n, 1]
#         temp = torch.zeros(max(num_ph, self.K_max), self.tseq.shape[-1])
#         temp[:num_ph, :] = torch.maximum(torch.minimum(self.tseq - birth, death - self.tseq), torch.tensor(0))    # shape: [n, T]
#         pl = torch.sort(temp, dim=0, descending=True).values[:self.K_max, :]    # shape: [K_max, T]
#         return pl