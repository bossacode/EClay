import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import gudhi as gd
import numpy as np
from scipy.stats import norm


class ECC(Function):
    @staticmethod
    def forward(ctx, x, func, steps, impulse):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W].
            func (function or method): Function that calculates ECC and gradient if needed.
            steps (int): Number of discretized points.
            impulse (float): Value used as approximation of dirac delta.

        Returns:
            torch.Tensor: Tensor of shape [B, C, steps].
        """
        backprop = x.requires_grad
        device = x.device
        ecc_collec = np.zeros((x.shape[0], x.shape[1], steps))
        grad_collec = np.zeros((x.shape[0], x.shape[1], x.shape[2]*x.shape[3], steps)) if backprop else None
        for b, data in enumerate(x.cpu().numpy()):  # iterate over batch
            for c, channel in enumerate(data):      # iterate over channel
                ecc, grad = func(channel, backprop)
                ecc_collec[b, c] = ecc
                if backprop:
                    grad_collec[b, c] = grad
        if backprop:
            ctx.save_for_backward(torch.from_numpy(grad_collec).to(device).to(torch.float32))
            ctx.input_size = x.shape
            ctx.impulse = impulse
        return torch.from_numpy(ecc_collec).to(device).to(torch.float32)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """_summary_

        Args:
            grad_output (torch.Tensor): Gradient w.r.t. to output. Tensor of shape [B, C, steps].

        Returns:
            _type_: _description_
        """
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_local, = ctx.saved_tensors                                     # shape: [B, C, H*W, steps]
            grad_input = torch.matmul(grad_local, grad_output.unsqueeze(-1))    # shape: shape: [B, C, H*W, 1]
            grad_input = grad_input.view(*ctx.input_size) * (-ctx.impulse)      # shape: shape: [B, C, H, W]
        return grad_input, None, None, None


class CubECC2d(nn.Module):
    def __init__(self, as_vertices=True, sublevel=True, size=[28, 28], interval=[0.02, 0.28], steps=32, scale=0.1):
        """
        Args:
            as_vertices (bool, optional): Use V-construction. If False, T-construction will be used. Defaults to True.
            sublevel (bool, optional): Use sublevel set filtration. If False, superlevel set filtration will be used by imposing a sublevel set filtration on negative data. Defaults to True.
            size (list, optional): [Height, Width] of image. Defaults to [28, 28].
            interval (list, optional): Minimum and maximum value of interval to consider. Defaults to [0.02, 0.28].
            steps (int, optional):  Number of discretized points. Defaults to 32.
            scale (float, optional): Standard deviation of Normal distribution controlling the magnitude of impulse that approximates the dirac delta function used for backpropagation. Defaults to 0.1.
        """
        super().__init__()
        self.as_vertices = as_vertices
        self.sublevel = sublevel
        self.h, self.w = size
        self.grid_h, self.grid_w = [2*i - 1 if as_vertices else 2*i + 1 for i in size]  # size of the cubical complex
        self.dimension = self._set_dimension()
        self.t_min, self.t_max = interval
        self.steps = steps
        self.resolution = (self.t_max - self.t_min) / (self.steps - 1)
        self.cal_ecc = self._cal_ecc_vtx if self.as_vertices else self._cal_ecc_topdim
        self.neighbor_idx_dict = self._return_vtx_i() if as_vertices else self._return_sq_i()   # dictionary mapping cell index to the indices of its neighboring vertices or squares
        self.pix_idx_dict = {(self._pix_i2vtx_i(i) if as_vertices else self._pix_i2sq_i(i)):i for i in range(size[0]*size[1])}  # dictionary mapping vertice or square index to corresponding pixel index in flattened image
        self.impulse = norm(loc=0, scale=scale).pdf(0)  # approximation of dirac delta used for backpropagtion
        self.lower_bound = self.t_min - self.resolution    # lower bound for skipping gradient calculation in backpropagation step
        
    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Tensor of shape [B, C, steps].
        """
        return ECC.apply(x, self.cal_ecc, self.steps, self.impulse)
    
    def _cal_ecc_vtx(self, x, backprop):
        """_summary_

        Args:
            x (numpy.ndarray): Numpy array of shape [H, W].
            backprop (bool): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        ecc = np.zeros(self.steps)
        grad_local = np.zeros([self.h*self.w, self.steps]) if backprop else None
        cub_cpx = gd.CubicalComplex(vertices=x if self.sublevel else -x)
        filtration = cub_cpx.all_cells().flatten()
        # for i, (dim, filt) in enumerate(zip(self.dimension, filtration)): # iterate over all simplexes in cubical complex
        #     if filt > self.t_max:
        #         continue
        idx = np.argsort(filtration)                    # index of filtration sorted in increasing order
        for i, dim, filt in zip(idx, self.dimension[idx], filtration[idx]): # iterate over all simplexes in cubical complex
            if filt > self.t_max:                       # stop when filtration value is beyond interval
                break
            t = max(np.ceil((filt - self.t_min) / self.resolution).astype(int), 0)
            ecc[t] += (-1)**dim

            # calculation of gradient only for inputs that require gradient
            if backprop:
                if filt < self.lower_bound:             # skip bc. gradient is 0 for simplices with filtration value under lower bound
                    continue
                # vertex
                if dim == 0:
                    pix_i = self.pix_idx_dict[i]        # index of the corresponding pixel in flattened original image
                    grad_local[pix_i, t] += 1
                # edge
                elif dim == 1:
                    vtx_i = self.neighbor_idx_dict[i]   # there are 2 neighbor vertices, vtx_i is list of 2 indexes
                    vtx_filt = filtration[vtx_i]        # filtration value of neighbor vertices
                    if vtx_filt[0] == vtx_filt[1]:      # split gradient when the neighboring vertices have the same filtration value
                        pix_i = [self.pix_idx_dict[i] for i in vtx_i]
                        grad_local[pix_i, [t, t]] -= 1/2
                    else:
                        pix_i = self.pix_idx_dict[vtx_i[np.argmax(vtx_filt)]]
                        grad_local[pix_i, t] -= 1
                # square
                else:
                    vtx_i = self.neighbor_idx_dict[i]   # there are 4 neighbor vertices, vtx_i is a np.array of 4 indexes
                    vtx_filt = filtration[vtx_i]
                    pix_i = [self.pix_idx_dict[i] for i in vtx_i[np.nonzero(vtx_filt == np.max(vtx_filt))[0]]]
                    grad_local[pix_i, [t]] += 1/len(pix_i)
        return np.cumsum(ecc), grad_local

    def _cal_ecc_topdim(self, x, backprop):
        """_summary_

        Args:
            x (numpy.ndarray): Numpy array of shape [H, W].
            backprop (bool): Whether or not the input requires gradient calculation

        Returns:
            _type_: _description_
        """
        ecc = np.zeros(self.steps)
        grad_local = np.zeros([self.h*self.w, self.steps]) if backprop else None
        cub_cpx = gd.CubicalComplex(top_dimensional_cells=x if self.sublevel else -x)
        filtration = cub_cpx.all_cells().flatten()
        # for i, (dim, filt) in enumerate(zip(self.dimension, filtration)): # iterate over all simplexes in cubical complex
        #     if filt > self.t_max:
        #         continue
        idx = np.argsort(filtration)                    # index of filtration sorted in increasing order
        for i, dim, filt in zip(idx, self.dimension[idx], filtration[idx]): # iterated over all simplexes in cubical complex
            if filt > self.t_max:                       # stop when filtration value is beyond interval
                break
            t = max(np.ceil((filt - self.t_min) / self.resolution).astype(int), 0)
            ecc[t] += (-1)**dim

            # calculation of gradient only for inputs that require gradient
            if backprop:
                if filt < self.lower_bound:             # skip bc. gradient is 0 for simplices with filtration value under lower bound
                    continue
                # square
                if dim == 2:
                    pix_i = self.pix_idx_dict[i]        # index of the corresponding pixel in flattened original image
                    grad_local[pix_i, t] += 1
                # edge
                elif dim == 1:
                    sq_i = self.neighbor_idx_dict[i]    # index of neighbor squares
                    if type(sq_i) == int:               # there is 1 neighbor square, sq_i is int denoting the index
                        pix_i = self.pix_idx_dict[sq_i]
                        grad_local[pix_i, t] -= 1
                    else:                               # there are 2 neighbor squares, sq_i is list of 2 indexes
                        sq_filt = filtration[sq_i]      # filtration value of neighbor squares
                        if sq_filt[0] == sq_filt[1]:    # split gradient when the neighboring squares have the same filtration value
                            pix_i = [self.pix_idx_dict[i] for i in sq_i]
                            grad_local[pix_i, [t, t]] -= 1/2
                        else:
                            pix_i = self.pix_idx_dict[sq_i[np.argmin(sq_filt)]]
                            grad_local[pix_i, t] -= 1
                # vertex
                else:
                    sq_i = self.neighbor_idx_dict[i]
                    if type(sq_i) == int:               # there is 1 neighbor square, sq_i is int
                        pix_i = self.pix_idx_dict[sq_i]
                        grad_local[pix_i, t] += 1
                    elif len(sq_i) == 2:                # there are 2 neighbor squares, sq_i is list of 2 indexes
                        sq_filt = filtration[sq_i]      # filtration value of neighbor squares
                        if sq_filt[0] == sq_filt[1]:    # split gradient when the neighboring squares have the same filtration value
                            pix_i = [self.pix_idx_dict[i] for i in sq_i]
                            grad_local[pix_i, [t, t]] += 1/2
                        else:
                            pix_i = self.pix_idx_dict[sq_i[np.argmin(sq_filt)]]
                            grad_local[pix_i, t] += 1
                    else:                               # there are 4 neighbor squares, sq_i is a np.array of 4 indexes
                        sq_filt = filtration[sq_i]
                        pix_i = [self.pix_idx_dict[i] for i in sq_i[np.nonzero(sq_filt == np.min(sq_filt))[0]]]
                        grad_local[pix_i, [t]] += 1/len(pix_i)
        return np.cumsum(ecc), grad_local
    
    def _pix_i2vtx_i(self, i):
        """
        Given index of a pixel in flattened image, this function returns index of the corresponding vertex in cubical complex. Used for V-constructed cubical complexes.

        Args:
            i (int): Index of a pixel in flatten image

        Returns:
            int: _description_
        """
        assert self.as_vertices, "Used for cubical complexes constructed using vertices"
        return (2*(i//self.w))*self.grid_w + 2*(i%self.w)
    
    def _pix_i2sq_i(self, i):
        """
        Given index of a pixel in flattened image, this function returns index of the corresponding square in cubical complex. Used for T-constructed cubical complexes.

        Args:
            i (int): _description_

        Returns:
            int: _description_
        """
        assert not self.as_vertices, "Used for cubical complexes constructed using top dimensional cells"
        return (2*(i//self.w) + 1)*self.grid_w + 2*(i%self.w) + 1

    def _return_vtx_i(self):
        """Returns a dictionary mapping the index of cells in cubical complex to the index of it neighboring vertices. Used for V-constructed cubical complexes.

        Returns:
            dict: _description_
        """
        assert self.as_vertices, "Used for cubical complexes constructed using vertices"
        vtx_i_dict = {}
        for dim, i in zip(self.dimension, range(self.grid_h*self.grid_w)):
            # vertex
            if dim == 0:
                continue
            # edge
            elif dim == 1:
                row_num = i // self.grid_w
                # even row
                if row_num % 2 == 0:
                    vtx_i = [i-1, i+1]  # index of neighbor vertices
                # odd row
                else:
                    vtx_i = [i-self.grid_w, i+self.grid_w]
            # square
            else:
                vtx_i = np.array([i-self.grid_w-1, i-self.grid_w+1, i+self.grid_w-1, i+self.grid_w+1])
            vtx_i_dict.update({i:vtx_i})
        return vtx_i_dict

    def _return_sq_i(self):
        """Returns a dictionary mapping the index of cells in cubical complex to the index of it neighboring squares. Used for T-constructed cubical complexes.

        Returns:
            dict: _description_
        """
        assert not self.as_vertices, "Used for cubical complexes constructed using top dimensional cells"
        sq_idx_dict = {}
        for dim, i in zip(self.dimension, range(self.grid_h*self.grid_w)):
            # square
            if dim == 2:
                continue
            # edge
            elif dim == 1:
                row_num = i // self.grid_w
                # even row
                if row_num % 2 == 0:
                    if row_num == 0:                        # top row
                        sq_i = i + self.grid_w              # index of neighbor square
                    elif row_num == self.grid_h - 1:        # bottom row
                        sq_i = i - self.grid_w
                    else:
                        sq_i = [i-self.grid_w, i+self.grid_w]
                # odd row
                else:
                    col_num = i % self.grid_w
                    if col_num == 0:                        # left-most column
                        sq_i = i + 1
                    elif col_num == self.grid_w - 1:        # right-most column
                        sq_i = i - 1
                    else:
                        sq_i = [i-1, i+1]
            # vertex
            else:
                row_num = i // self.grid_w
                col_num = i % self.grid_w
                # top row
                if row_num == 0:
                    if i == 0:                              # top left corner
                        sq_i = i + self.grid_w + 1
                    elif i == self.grid_w - 1:              # top right corner
                        sq_i = i + self.grid_w - 1
                    else:
                        sq_i = [i+self.grid_w-1, i+self.grid_w+1]
                # bottom row
                elif row_num == self.grid_h - 1:
                    if i == self.grid_w * (self.grid_h-1):  # bottom left corner
                        sq_i = i - self.grid_w + 1
                    elif i == self.grid_w*self.grid_h - 1:  # bottom right corner
                        sq_i = i - self.grid_w - 1
                    else:
                        sq_i = [i-self.grid_w-1, i-self.grid_w+1]
                # left-most column
                elif col_num == 0:                      
                    sq_i = [i+1-self.grid_w, i+1+self.grid_w]
                # right-most column
                elif col_num == self.grid_w - 1:
                    sq_i = [i-1-self.grid_w, i-1+self.grid_w]
                else:
                    sq_i = np.array([i-self.grid_w-1, i-self.grid_w+1, i+self.grid_w-1, i+self.grid_w+1])
            sq_idx_dict.update({i:sq_i})
        return sq_idx_dict

    def _set_dimension(self):
        """
        Sets dimension for all cubes in the cubical complex. Dimensions of vertice, edge, square are 0, 1, 2 respectively. Even rows consist of (vertice, edge, vertice, edge, ..., vertice, edge, vertice) and odd rows consist of (edge, square, edge, square, ..., edge, square, edge).

        Returns:
            _type_: _description_
        """
        dimension = np.zeros([self.grid_h, self.grid_w])
        dimension[[i for i in range(self.grid_h) if i % 2 == 1], :] += 1
        dimension[:, [i for i in range(self.grid_w) if i % 2 == 1]] += 1
        return dimension.flatten()


class GThetaEC(nn.Module):
    def __init__(self, num_features=[32, 32]):
        """
        Args:
            num_features (list, optional): List containing the size of each layer. Defaults to [32, 32].
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


class ECLay(nn.Module):
    def __init__(self, as_vertices=True, sublevel=True, size=[28, 28], interval=[0.02, 0.28], steps=32, in_channels=1, hidden_features=[32], scale=0.1,):
        """_summary_

        Args:
            as_vertices (bool, optional): _description_. Defaults to True.
            sublevel (bool, optional): _description_. Defaults to True.
            size (list, optional): _description_. Defaults to [28, 28].
            interval (list, optional): _description_. Defaults to [0.02, 0.28].
            steps (int, optional): _description_. Defaults to 32.
            in_channels (int, optional): _description_. Defaults to 1.
            hidden_features (list, optional): _description_. Defaults to [32].
            scale (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.ec_layer = CubECC2d(as_vertices, sublevel, size, interval, steps, scale)
        self.flatten = nn.Flatten()
        self.gtheta = GThetaEC([in_channels * steps] + hidden_features)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, C, H, W]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        ec = self.flatten(self.ec_layer(x)) # shape: [batch_size, (C * steps)]
        return self.gtheta(ec)