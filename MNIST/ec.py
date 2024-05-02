import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import gudhi as gd
import numpy as np
from scipy.stats import norm
# import matplotlib.pyplot as plt


class EccGrad(Function):
    @staticmethod
    def forward(ctx, x, func, resolution, impulse):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W]
            func (_type_): _description_
            resolution (int): _description_
            impulse (float): _description_

        Returns:
            torch.Tensor: Tensor of shape [B, C, T]
        """
        backprop = x.requires_grad
        device = x.device
        x = x.cpu()
        ecc_collec = np.zeros((x.shape[0], x.shape[1], resolution))
        grad_collec = np.zeros((x.shape[0], x.shape[1], x.shape[2]*x.shape[3], resolution)) if backprop else None
        for b, data in enumerate(x):              # iterate over batch
            for c, channel in enumerate(data):    # iterate over channel
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
            grad_output (torch.Tensor): Gradient w.r.t. to output. Tensor of shape [B, C, T]

        Returns:
            _type_: _description_
        """
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_local, = ctx.saved_tensors                                     # shape: [B, C, H*W, T]
            grad_input = torch.matmul(grad_local, grad_output.unsqueeze(-1))    # shape: shape: [B, C, H*W, 1]
            grad_input = grad_input.view(*ctx.input_size) * (-ctx.impulse)      # shape: shape: [B, C, H, W]
            # plt.imshow(grad_input[0].squeeze(), cmap="gray")
            # plt.show();
        return grad_input, None, None, None


class CubEcc2d(nn.Module):
    def __init__(self, as_vertices=True, sublevel=True, size=[28, 28], interval=[0, 7, 32]):
        """_summary_

        Args:
            as_vertices (bool, optional): Set pixels in the image as filtration values of vertices in cubical complex. If False, pixels will be set as filtration values of top dimensional cells. Defaults to True.
            sublevel (bool, optional): Use sublevel set filtration. If False, superlevel set filtration will be used by imposing a sublevel set filtration on data multiplied by -1. Defaults to True.
            size (list, optional): [Height, Width] of image. Defaults to [28, 28].
            interval (list, optional): _description_. Defaults to [0, 7, 32].
        """
        super().__init__()
        self.as_vertices = as_vertices
        self.sublevel = sublevel
        self.h, self.w = size
        self.grid_h, self.grid_w = [2*i - 1 if as_vertices else 2*i + 1 for i in size]  # size of the cubical complex
        self.dimension = self._set_dimension()
        self.t_min, self.t_max, self.resolution = interval
        self.scale = (self.resolution - 1) / (self.t_max - self.t_min)
        self.cal_ecc = self._cal_ecc_vtx if self.as_vertices else self._cal_ecc_topdim
        self.idx_dict = {self._pix_i2vtx_i(i):i for i in range(size[0]*size[1])} if as_vertices else {self._pix_i2sq_i(i):i for i in range(size[0]*size[1])}
        self.impulse = norm(loc=0, scale=0.1).pdf(0)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Tensor of shape [B, C, T]
        """
        return EccGrad.apply(x, self.cal_ecc, self.resolution, self.impulse)

    
    def _cal_ecc_vtx(self, x, backprop):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [H, W]
            backprop (bool): Whether or not the input requires gradient calculation

        Returns:
            _type_: _description_
        """
        ecc = np.zeros(self.resolution)
        grad_local = np.zeros([self.h*self.w, self.resolution]) if backprop else None

        cub_cpx = gd.CubicalComplex(vertices=x) if self.sublevel else gd.CubicalComplex(vertices=-x)
        filtration = cub_cpx.all_cells().flatten()
        idx = np.argsort(filtration)    # index of filtration sorted in increasing order
        for i, dim, filt in zip(idx, self.dimension[idx], filtration[idx]): # iterated over all simplexes in cubical complex
            if filt > self.t_max:       # stop when filtration value is beyond interval
                break
            t = max(np.ceil((filt - self.t_min)*self.scale).astype(int), 0)
            ecc[t] += (-1)**dim

            # calculation of gradient only for inputs that require gradient
            if backprop:
                # vertex
                if dim == 0:
                    pix_i = self.idx_dict[i]        # index of the corresponding pixel in flattened original image
                    grad_local[pix_i, t] += 1
                # edge
                elif dim == 1:
                    row_num = i // self.grid_w
                    # even row
                    if row_num % 2 == 0:
                        vtx_i = [i-1, i+1]          # index of neighbor vertices
                    # odd row
                    else:
                        vtx_i = [i-self.grid_w, i+self.grid_w]
                    vtx_filt = filtration[vtx_i]    # filtration value of neighbor vertices
                    if vtx_filt[0] == vtx_filt[1]:  # split gradient when the neighboring vertices have the same filtration value
                        pix_i = [self.idx_dict[i] for i in vtx_i]
                        grad_local[pix_i, [t, t]] -= 1/2
                    else:
                        pix_i = self.idx_dict[vtx_i[np.argmax(vtx_filt)]]
                        grad_local[pix_i, t] -= 1
                # square
                else:
                    vtx_i = np.array([i-self.grid_w-1, i-self.grid_w+1, i+self.grid_w-1, i+self.grid_w+1])
                    vtx_filt = filtration[vtx_i]
                    pix_i = [self.idx_dict[i] for i in vtx_i[np.nonzero(vtx_filt == np.max(vtx_filt))[0]]]
                    grad_local[pix_i, [t]] += 1/len(pix_i)
        return np.cumsum(ecc), grad_local

    def _cal_ecc_topdim(self, x, backprop):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [H, W]
            backprop (bool): Whether or not the input requires gradient calculation

        Returns:
            _type_: _description_
        """
        ecc = np.zeros(self.resolution)
        grad_local = np.zeros([self.h*self.w, self.resolution]) if backprop else None
        
        cub_cpx = gd.CubicalComplex(top_dimensional_cells=x) if self.sublevel else gd.CubicalComplex(top_dimensional_cells=-x)
        filtration = cub_cpx.all_cells().flatten()
        idx = np.argsort(filtration)    # index of filtration sorted in increasing order
        for i, dim, filt in zip(idx, self.dimension[idx], filtration[idx]): # iterated over all simplexes in cubical complex
            if filt > self.t_max:       # stop when filtration value is beyond interval
                break
            t = max(np.ceil((filt - self.t_min)*self.scale).astype(int), 0)
            ecc[t] += (-1)**dim

            # calculation of gradient only for inputs that require gradient
            if backprop:
                # square
                if dim == 2:
                    pix_i = self.idx_dict[i]                    # index of the corresponding pixel in flattened original image
                    grad_local[pix_i, t] += 1
                # edge
                elif dim == 1:
                    row_num = i // self.grid_w
                    # even row
                    if row_num % 2 == 0:
                        if row_num == 0:                        # top row
                            sq_i = i + self.grid_w              # index of neighbor square
                            pix_i = self.idx_dict[sq_i]
                            grad_local[pix_i, t] -= 1
                        elif row_num == self.grid_h - 1:        # bottom row
                            sq_i = i - self.grid_w
                            pix_i = self.idx_dict[sq_i]
                            grad_local[pix_i, t] -= 1
                        else:
                            sq_i = [i-self.grid_w, i+self.grid_w]
                            sq_filt = filtration[sq_i]          # filtration value of neighbor squares
                            if sq_filt[0] == sq_filt[1]:        # split gradient when the neighboring squares have the same filtration value
                                pix_i = [self.idx_dict[i] for i in sq_i]
                                grad_local[pix_i, [t, t]] -= 1/2
                            else:
                                pix_i = self.idx_dict[sq_i[np.argmin(sq_filt)]]
                                grad_local[pix_i, t] -= 1
                    # odd row
                    else:
                        col_num = i % self.grid_w
                        if col_num == 0:                        # left-most column
                            sq_i = i + 1
                            pix_i = self.idx_dict[sq_i]
                            grad_local[pix_i, t] -= 1
                        elif col_num == self.grid_w - 1:        # right-most column
                            sq_i = i - 1
                            pix_i = self.idx_dict[sq_i]
                            grad_local[pix_i, t] -= 1
                        else:
                            sq_i = [i-1, i+1]
                            sq_filt = filtration[sq_i]
                            if sq_filt[0] == sq_filt[1]:        # split gradient when the neighboring squares have the same filtration value
                                pix_i = [self.idx_dict[i] for i in sq_i]
                                grad_local[pix_i, [t, t]] -= 1/2
                            else:
                                pix_i = self.idx_dict[sq_i[np.argmin(sq_filt)]]
                                grad_local[pix_i, t] -= 1
                # vertex
                else:
                    row_num = i // self.grid_w
                    col_num = i % self.grid_w
                    # top row
                    if row_num == 0:
                        if i == 0:                              # top left corner
                            sq_i = i + self.grid_w + 1
                            pix_i = self.idx_dict[sq_i]
                            grad_local[pix_i, t] += 1
                        elif i == self.grid_w - 1:              # top right corner
                            sq_i = i + self.grid_w - 1
                            pix_i = self.idx_dict[sq_i]
                            grad_local[pix_i, t] += 1
                        else:
                            sq_i = [i+self.grid_w-1, i+self.grid_w+1]
                            sq_filt = filtration[sq_i]
                            if sq_filt[0] == sq_filt[1]:        # split gradient when the neighboring squares have the same filtration value
                                pix_i = [self.idx_dict[i] for i in sq_i]
                                grad_local[pix_i, [t, t]] += 1/2
                            else:
                                pix_i = self.idx_dict[sq_i[np.argmin(sq_filt)]]
                                grad_local[pix_i, t] += 1
                    # bottom row
                    elif row_num == self.grid_h - 1:
                        if i == self.grid_w * (self.grid_h-1):  # bottom left corner
                            sq_i = i - self.grid_w + 1
                            pix_i = self.idx_dict[sq_i]
                            grad_local[pix_i, t] += 1
                        elif i == self.grid_w*self.grid_h - 1:  # bottom right corner
                            sq_i = i - self.grid_w - 1
                            pix_i = self.idx_dict[sq_i]
                            grad_local[pix_i, t] += 1
                        else:
                            sq_i = [i-self.grid_w-1, i-self.grid_w+1]
                            sq_filt = filtration[sq_i]
                            if sq_filt[0] == sq_filt[1]:        # split gradient when the neighboring squares have the same filtration value
                                pix_i = [self.idx_dict[i] for i in sq_i]
                                grad_local[pix_i, [t, t]] += 1/2
                            else:
                                pix_i = self.idx_dict[sq_i[np.argmin(sq_filt)]]
                                grad_local[pix_i, t] += 1
                    # left-most column
                    elif col_num == 0:                      
                        sq_i = [i+1-self.grid_w, i+1+self.grid_w]
                        sq_filt = filtration[sq_i]
                        if sq_filt[0] == sq_filt[1]:            # split gradient when the neighboring squares have the same filtration value
                            pix_i = [self.idx_dict[i] for i in sq_i]
                            grad_local[pix_i, [t, t]] += 1/2
                        else:
                            pix_i = self.idx_dict[sq_i[np.argmin(sq_filt)]]
                            grad_local[pix_i, t] += 1
                    # right-most column
                    elif col_num == self.grid_w - 1:
                        sq_i = [i-1-self.grid_w, i-1+self.grid_w]
                        sq_filt = filtration[sq_i]
                        if sq_filt[0] == sq_filt[1]:            # split gradient when the neighboring squares have the same filtration value
                            pix_i = [self.idx_dict[i] for i in sq_i]
                            grad_local[pix_i, [t, t]] += 1/2
                        else:
                            pix_i = self.idx_dict[sq_i[np.argmin(sq_filt)]]
                            grad_local[pix_i, t] += 1
                    else:
                        sq_i = np.array([i-self.grid_w-1, i-self.grid_w+1, i+self.grid_w-1, i+self.grid_w+1])
                        sq_filt = filtration[sq_i]
                        pix_i = [self.idx_dict[i] for i in sq_i[np.nonzero(sq_filt == np.min(sq_filt))[0]]]
                        grad_local[pix_i, [t]] += 1/len(pix_i)
        return np.cumsum(ecc), grad_local
    
    def _pix_i2vtx_i(self, i):
        """
        Given index of a pixel in flattened image, this function returns index of the corresponding vertex in flattened cubical complex. Should be used for cubical complexes constructed using vertices.

        Args:
            i (int): Index of a pixel in flatten image

        Returns:
            int: _description_
        """
        return (2*(i//self.w))*self.grid_w + 2*(i%self.w)
    
    def _pix_i2sq_i(self, i):
        """
        Given index of a pixel in flattened image, this function returns index of the corresponding square in flattened cubical complex. Should be used for cubical complexes constructed using top dimensional cells.

        Args:
            i (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (2*(i//self.w) + 1)*self.grid_w + 2*(i%self.w) + 1

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


class ECLay(nn.Module):
    def __init__(self, as_vertices=True, sublevel=True, size=[28, 28], interval=[0, 7, 32], in_channels=1, hidden_features=[64, 32]):
        """_summary_

        Args:
            superlevel (bool, optional): _description_. Defaults to False.
            tseq (list, optional): _description_. Defaults to [0, 7, 32].
            in_channels (int, optional): _description_. Defaults to 1.
            hidden_features (list, optional): _description_. Defaults to [64, 32].
        """
        super().__init__()
        self.ec_layer = CubEcc2d(as_vertices, sublevel, size, interval)
        self.flatten = nn.Flatten()
        self.gtheta_layer = self._make_gtheta_layer(in_channels * interval[-1], hidden_features)

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