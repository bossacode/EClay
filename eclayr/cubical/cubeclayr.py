import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math
from eclayr.cubical.cython_eclayr.ecc import EccBackbone, EccBackbone3d
from eclayr.cubical.cython_sigmoid.ecc_sigmoid import SigEccBackbone, SigEccBackbone3d


class EccFunction(Function):
    @staticmethod
    def forward(ctx, x, func, device):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W].
            func (function or method): Function that calculates ECC and gradient if necessary.
            device (str, optional): Device to which the output tensor will be moved.

        Returns:
            torch.Tensor: Tensor of shape [B, C, steps].
        """
        backprop = x.requires_grad
        ecc, grad = func(x.cpu().numpy(), backprop)
        if backprop:
            ctx.save_for_backward(torch.from_numpy(grad).to(device))
            ctx.input_shape = x.shape
        return torch.from_numpy(ecc).to(device)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """_summary_

        Args:
            grad_out (torch.Tensor): Gradient w.r.t. to output. Tensor of shape [B, C, steps].

        Returns:
            _type_: _description_
        """
        grad_in = None
        if ctx.needs_input_grad[0]:
            grad_local, = ctx.saved_tensors                             # shape: [B, C, H*W, steps]
            grad_in = torch.matmul(grad_local, grad_out.unsqueeze(-1))  # shape: shape: [B, C, H*W, 1]
            grad_in = grad_in.view(*ctx.input_shape)                    # shape: shape: [B, C, H, W]
        return grad_in, None, None


class CubEclayr(nn.Module):
    def __init__(self, interval=[0., 1.], steps=32, constr="V", sublevel=True, beta=0.1, postprocess=nn.Identity(), device="cuda" if torch.cuda.is_available() else "cpu",
                 *args, **kwargs):
        """
        Args:            
            interval (Iterable[float], optional): Minimum and maximum value of interval to consider. Defaults to [0., 1.].
            steps (int, optional):  Number of discretized points. Defaults to 32.
            constr (str, optional): One of V or T, corresponding to V-construction and T-construction, respectively. Defaults to V.
            sublevel (bool, optional): Whether to use sublevel set filtration. If False, superlevel set filtration will be used. Defaults to True.
            beta (float, optional): Controls the magnitude of impulse that approximates the dirac delta function used for backpropagation. Smaller values yield higher impulse. Defaults to 0.1.
            postprocess (_type_, optional): _description_. Defaults to nn.Identity().
            device (str, optional): Device to which the output tensor will be moved. Defaults to "cuda" if available, otherwise "cpu".
        """
        assert len(interval) == 2, "Interval must consist of two values."
        assert interval[1] > interval[0], "End point of interval must be larger than starting point of interval."
        assert steps > 1, "Number of steps should be larger than 1."
        assert constr == "V" or constr == "T", "Construction must be one of V or T."

        super().__init__()
        self.interval = interval if sublevel else [-i for i in reversed(interval)] # change interval when superlevel set filtration is used
        self.steps = steps
        self.constr = constr
        self.sublevel = sublevel
        self.impulse = 1 / (abs(beta) * math.sqrt(math.pi))
        self.flatten = nn.Flatten()
        self.postprocess = postprocess
        self.device = device
        self.func = None
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Tensor of shape [B, C, steps].
        """
        if self.func is None:   # lazily initialize when first batch is passed
            dim = len(x.shape[2:])
            if dim == 2:
                ecc = EccBackbone(x.shape[2:], self.interval, self.steps, self.constr, self.impulse)
            elif dim == 3:
                ecc = EccBackbone3d(x.shape[2:], self.interval, self.steps, self.constr, self.impulse)
            else:
                raise NotImplementedError
            self.func = ecc.cal_ecc_vtx if self.constr=="V" else ecc.cal_ecc_topdim     # function that calculates ECC and gradient if necessary.

        x = x if self.sublevel else -x  # impose sublevel set filtration on negative data when superlevel set filtration is used
        x = EccFunction.apply(x, self.func, self.device)
        x = self.flatten(x)
        x = self.postprocess(x)
        return x


class SigCubEclayr(nn.Module):
    def __init__(self, interval=[0., 1.], steps=32, constr="V", sublevel=True, lam=200, postprocess=nn.Identity(), device="cuda" if torch.cuda.is_available() else "cpu",
                 *args, **kwargs):
        """
        Args:            
            interval (Iterable[float], optional): Minimum and maximum value of interval to consider. Defaults to [0., 1.].
            steps (int, optional):  Number of discretized points. Defaults to 32.
            constr (str, optional): One of V or T, corresponding to V-construction and T-construction, respectively. Defaults to V.
            sublevel (bool, optional): Whether to use sublevel set filtration. If False, superlevel set filtration will be used. Defaults to True.
            lam (float, optional): Controls the tightness of sigmoid approximation. Defaults to 200.
            postprocess (_type_, optional): _description_. Defaults to nn.Identity().
        """
        assert len(interval) == 2, "Interval must consist of two values."
        assert interval[1] > interval[0], "End point of interval must be larger than starting point of interval."
        assert steps > 1, "Number of steps should be larger than 1."
        assert constr == "V" or constr == "T", "Construction must be one of V or T."

        super().__init__()
        self.interval = interval if sublevel else [-i for i in reversed(interval)] # change interval when superlevel set filtration is used
        self.steps = steps
        self.constr = constr
        self.sublevel = sublevel
        self.lam = lam
        self.flatten = nn.Flatten()
        self.postprocess = postprocess
        self.device = device
        self.func = None
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Tensor of shape [B, C, steps].
        """
        if self.func is None:   # lazily initialize when first batch is passed
            dim = len(x.shape[2:])
            if dim == 2:
                ecc = SigEccBackbone(x.shape[2:], self.interval, self.steps, self.constr, self.lam)
            elif dim == 3:
                ecc = SigEccBackbone3d(x.shape[2:], self.interval, self.steps, self.constr, self.lam)
            else:
                raise NotImplementedError
            self.func = ecc.cal_ecc_vtx if self.constr=="V" else ecc.cal_ecc_topdim     # function that calculates ECC and gradient if necessary.

        x = x if self.sublevel else -x  # impose sublevel set filtration on negative data when superlevel set filtration is used
        x = EccFunction.apply(x, self.func, self.device)
        x = self.flatten(x)
        x = self.postprocess(x)
        return x