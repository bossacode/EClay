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
            x (torch.Tensor): Point cloud of shape [B, D].
            func (_type_): _description_
            resolution (_type_): _description_
            impulse (float): _description_

        Returns:
            torch.Tensor: Tensor of shape [steps, ]
        """
        backprop = x.requires_grad
        device = x.device
        ecc, grad = func(x.cpu().numpy(), backprop)
        if backprop:
            ctx.save_for_backward(torch.from_numpy(grad).to(device).to(torch.float32))
            ctx.impulse = impulse
        return torch.from_numpy(ecc).to(device).to(torch.float32)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """_summary_

        Args:
            grad_output (torch.Tensor): Gradient w.r.t. to output. Tensor of shape [steps, ]

        Returns:
            _type_: _description_
        """
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_local, = ctx.saved_tensors                         # shape: [B, D, steps]
            grad_input = torch.matmul(grad_local, grad_output)      # shape: [B, D]
            grad_input = grad_input * (-ctx.impulse)
        return grad_input, None, None, None


class RipsECC(nn.Module):
    def __init__(self, interval=[0, 2], steps=64, scale=0.1):
        super().__init__()
        self.t_min, self.t_max = interval
        self.steps = steps
        self.scale = (self.steps - 1) / (self.t_max - self.t_min)
        self.impulse = norm(loc=0, scale=scale).pdf(0)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, D].

        Returns:
            _type_: Tensor of shape [B, T]. B is batch size and T is resolution.
        """
        return ECC.apply(x, self._cal_ecc, self.steps, self.impulse)

    def _cal_ecc(self, x , backprop):
        """_summary_

        Args:
            x (numpy.ndarray): Point cloud of shape [B, D]
            backprop (bool): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        ecc = np.zeros(self.steps)
        grad_local = np.zeros([*x.shape, self.steps]) if backprop else None    # shape: [B, D, T]

        skeleton = gd.RipsComplex(points = x, max_edge_length=self.t_max)
        st = skeleton.create_simplex_tree(max_dimension=1)
        for vtx_idx, filt in st.get_filtration():
            dim = len(vtx_idx) - 1     # dimension of simplex
            t = max(np.ceil((filt - self.t_min)*self.scale).astype(int), 0)
            ecc[t] += (-1)**dim

            # calculation of gradient only for inputs that require gradient
            if backprop:
                # vertex
                if dim == 0:
                    continue
                # edge
                elif dim == 1:
                    vtx_1, vtx_2 = x[vtx_idx]
                    grad = np.stack([(vtx_1 - vtx_2)/filt, (vtx_2 - vtx_1)/filt], axis=0)
                    grad_local[vtx_idx, :, t] -= grad
                # triangle or higher dimensional simplex
                else:
                    raise NotImplementedError("Backpropagation not implemented for 3-simplex and higher")
        return np.cumsum(ecc), grad_local


class AlphaECC(nn.Module):
    def __init__(self, interval=[0, 1], steps=64, scale=0.1):
        super().__init__()
        self.t_min, self.t_max = interval
        self.steps = steps
        self.scale = (self.steps-1) / (self.t_max-self.t_min)
        self.impulse = norm(loc=0, scale=scale).pdf(0)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, D].

        Returns:
            _type_: Tensor of shape [steps, ].
        """
        return ECC.apply(x, self._cal_ecc, self.steps, self.impulse)

    def _cal_ecc(self, x, backprop):
        """_summary_

        Args:
            x (numpy.ndarray): Point cloud of shape [B, D]
            backprop (bool): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        ecc = np.zeros(self.steps)
        grad_local = np.zeros([*x.shape, self.steps]) if backprop else None    # shape: [B, D, T]
        prev_filt, prev_vtx_idx, prev_grad = None, None, None

        skeleton = gd.AlphaComplex(points = x)
        st = skeleton.create_simplex_tree()
        for vtx_idx, filt in reversed(list(st.get_filtration())):  # decreasing order
            if filt > self.t_max:
                continue
            dim = len(vtx_idx) - 1     # dimension of simplex
            t = max(np.ceil((filt - self.t_min)*self.scale).astype(int), 0)
            ecc[t] += (-1)**dim
            
            # calculation of gradient only for inputs that require gradient
            if backprop:
                # vertex
                if dim == 0:
                    continue
                # edge
                elif dim == 1:
                    if filt == prev_filt:   # attached by higher dimensional simplex
                        ind = [i in vtx_idx for i in prev_vtx_idx]
                        grad = prev_grad[ind]
                        assert len(grad) != 0, "Violation of Alpha general position assumption"
                        grad_local[vtx_idx, :, t] -= grad
                    else:                   # attaching simplex
                        vtx_1, vtx_2 = x[vtx_idx]
                        grad = np.stack([(vtx_1 - vtx_2)/2, (vtx_2 - vtx_1)/2], axis=0)
                        grad_local[vtx_idx, :, t] -= grad
                # triangle
                elif dim == 2:
                    vtx_1, vtx_2, vtx_3 = x[vtx_idx]
                    grad = np.stack([self._grad_u(vtx_1, vtx_2, vtx_3), self._grad_v(vtx_1, vtx_2, vtx_3), self._grad_w(vtx_1, vtx_2, vtx_3)], axis=0)
                    grad_local[vtx_idx, :, t] += grad
                    prev_filt = filt
                    prev_vtx_idx = vtx_idx
                    prev_grad = grad
                # tetrahedron or higher dimensional simplex
                else:
                    raise NotImplementedError("Backpropagation not implemented for 3-simplex and higher")
        return np.cumsum(ecc), grad_local
    

    @staticmethod
    def _grad_u(u, v, w):
        """Given 3 points  that form a triangle in Alpha Complex, compute the gradient w.r.t. u.

        Args:
            u (_type_, optional): _description_.
            v (_type_, optional): _description_.
            w (_type_, optional): _description_.

        Returns:
            _type_: _description_
        """
        t_0 = (u - v)
        t_1 = (np.linalg.norm(t_0) ** 2)
        t_2 = np.linalg.norm((v - w))
        t_3 = (t_2 ** 2)
        t_4 = (u - w)
        t_5 = (np.linalg.norm(t_4) ** 2)
        t_6 = (4 * t_1)
        t_7 = ((t_1 + t_3) - t_5)
        t_8 = ((t_6 * t_3) - (t_7 ** 2))
        t_9 = (t_8 ** 2)
        gradient = ((((((2 * t_3) * t_5) / t_8) * t_0) + ((((2 * t_1) * t_3) / t_8) * t_4)) - (((((8 * t_1) * ((t_2 ** 4) * t_5)) / t_9) * t_0) - ((((t_6 * ((t_3 * t_5) * t_7)) / t_9) * t_0) - (((4 * ((t_5 * t_1) * (t_3 * t_7))) / t_9) * t_4))))
        return gradient

    @staticmethod
    def _grad_v(u, v, w):
        """Given 3 points  that form a triangle in Alpha Complex, compute the gradient w.r.t. v.

        Args:
            u (_type_, optional): _description_. Defaults to u.
            v (_type_, optional): _description_. Defaults to v.
            w (_type_, optional): _description_. Defaults to w.

        Returns:
            _type_: _description_
        """
        t_0 = (u - v)
        t_1 = np.linalg.norm(t_0)
        t_2 = (t_1 ** 2)
        t_3 = (v - w)
        t_4 = np.linalg.norm(t_3)
        t_5 = (t_4 ** 2)
        t_6 = (np.linalg.norm((u - w)) ** 2)
        t_7 = ((t_2 + t_5) - t_6)
        t_8 = (((4 * t_2) * t_5) - (t_7 ** 2))
        t_9 = (t_8 ** 2)
        t_10 = (t_2 * t_5)
        gradient = ((((((2 * t_2) * t_6) / t_8) * t_3) - (((2 * (t_5 * t_6)) / t_8) * t_0)) - ((((((8 * t_5) * ((t_1 ** 4) * t_6)) / t_9) * t_3) - (((8 * ((t_2 * (t_4 ** 4)) * t_6)) / t_9) * t_0)) - (((((4 * t_5) * ((t_2 * t_6) * t_7)) / t_9) * t_3) - (((4 * (t_10 * (t_6 * t_7))) / t_9) * t_0))))
        return gradient
    
    @staticmethod
    def _grad_w(u, v, w):
        """Given 3 points  that form a triangle in Alpha Complex, compute the gradient w.r.t. w.

        Args:
            u (_type_): _description_
            v (_type_): _description_
            w (_type_): _description_

        Returns:
            _type_: _description_
        """
        t_0 = np.linalg.norm((u - v))
        t_1 = (t_0 ** 2)
        t_2 = (v - w)
        t_3 = (np.linalg.norm(t_2) ** 2)
        t_4 = (u - w)
        t_5 = (np.linalg.norm(t_4) ** 2)
        t_6 = ((t_1 + t_3) - t_5)
        t_7 = (((4 * t_1) * t_3) - (t_6 ** 2))
        t_8 = (2 * t_1)
        t_9 = (t_1 * t_3)
        t_10 = (t_7 ** 2)
        gradient = -(((((t_8 * t_5) / t_7) * t_2) + (((t_8 * t_3) / t_7) * t_4)) - ((((((8 * t_3) * ((t_0 ** 4) * t_5)) / t_10) * t_2) + ((((4 * t_5) * (t_9 * t_6)) / t_10) * t_4)) - (((4 * ((t_3 * t_1) * (t_5 * t_6))) / t_10) * t_2)))
        return gradient


class GTheta(nn.Module):
    def __init__(self, num_features=[64, 32]):
        """
        Args:
            num_features (list, optional): List containing the size of each layer. Defaults to [64, 32].
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
    def __init__(self, interval=[0, 1], steps=64, hidden_features=[32], scale=0.1, type="Rips"):
        """_summary_

        Args:
            interval (list, optional): _description_. Defaults to [0, 1].
            steps (int, optional): _description_. Defaults to 64.
            hidden_features (list, optional): _description_. Defaults to [32].
            scale (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.ec_layer = RipsECC(interval, steps, scale) if type == "Rips" else AlphaECC(interval, steps, scale)
        self.gtheta = GTheta([steps] + hidden_features)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, D]

        Returns:
            output: Tensor of shape [out_features]
        """
        ec = self.ec_layer(x) # shape: [steps]
        return self.gtheta(ec)