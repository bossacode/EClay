import torch
import torch.nn as nn


def make_grid(lims=[[1,28], [1,28]], size=[28, 28]):
    """
    Creates a tensor of 2D grid points.
    Grid points have one-to-one correspondence with input pixel values that are flattened in row-major order.

    Args:
        lims: list or tuple in the form of [[domain of H], [domain of W]]
        size: list or tuple in the form of [H, W]
    Returns:
        grid: Tensor of shape [(H*W), 2]
    """
    assert len(size) == 2 and len(lims) == len(size)
    expansions = [torch.linspace(end, start, steps) if i == 0 else torch.linspace(start, end, steps) for i, ((start, end), steps) in enumerate(zip(lims, size))]
    grid = torch.index_select(torch.cartesian_prod(*expansions),
                        dim=1,
                        index=torch.tensor([1,0]))
    return grid


def cal_dist(grid, r=2):
    """
    Calculate distance between all cooridnate points on grid.

    Args:
        grid: Tensor of shape [(H*W), 2]
        r:
    Returns:
        distance: Tensor of shape [(H*W), (H*W)]
    """
    X = grid.unsqueeze(1)
    Y = grid.unsqueeze(0)
    if r == 2:
        dist = torch.sqrt(torch.sum((X - Y)**2, -1))
    elif r == 1:
        dist = torch.sum(torch.abs(X - Y), -1)
    else:
        dist = torch.pow(torch.sum((X - Y)**r, -1), 1/r)
    return dist


def dtm_using_knn(knn_dist, knn_index, input, bound, r=2):
    """
    Weighted DTM using KNN.

    Args:
        knn_dist: Tensor of shape [(H*W), max_k]
        knn_index: Tensor of shape [(H*W), max_k]
        input: Tensor of shape [batch_size, C, (H*W)]         # grad
        bound: Tensor of shape [batch_size, C, 1]             # grad
        r: Int r-Norm

    Returns:
        dtm_val: Tensor of shape [batch_size, C, (H*W)]
    """
    batch_size = input.shape[0]
    C = input.shape[1]
    HW = input.shape[-1]
    
    input_temp = input.unsqueeze(2).expand(-1, -1, HW, -1)                  # shape: [batch_size, C, (H*W), (H*W)]
    knn_index = knn_index.view(1, 1, HW, -1).expand(batch_size, C, -1, -1)  # shape: [batch_size, C, (H*W), k]
    knn_weight = torch.gather(input_temp, -1, knn_index)                    # shape: [batch_size, C, (H*W), k]    

    # finding k's s.t. sum({Wi: Wi in (k-1)-NN}) < m0*sum({Wi: i=1...n}) <= sum({Wi: Wi in k-NN})
    cum_knn_weight = knn_weight.cumsum(-1)                                  # shape: [batch_size, C, (H*W), k]
    bound = bound.unsqueeze(-1)                                             # shape: [batch_size, C, 1, 1]
    k = torch.searchsorted(cum_knn_weight, bound.repeat(1, 1, HW, 1))       # shape: [batch_size, C, (H*W), 1]
    
    # prevent index out of bounds error when some values of k_index equal HW
    if (k == HW).any():
        k[k == HW] -= 1

    if r == 2:
        r_dist = knn_dist.square().view(1, 1, HW, -1).expand(batch_size, C, -1, -1) # shape: [batch_size, C, (H*W), k]
        cum_dist = torch.cumsum(r_dist * knn_weight, -1)                            # shape: [batch_size, C, (H*W), k]
        dtm_val = torch.gather(cum_dist + r_dist*(bound-cum_knn_weight), -1, k)     # shape: [batch_size, C, (H*W), 1]
        dtm_val = torch.sqrt(dtm_val/bound)
    elif r == 1:
        r_dist = knn_dist.view(1, 1, HW, -1).expand(batch_size, C, -1, -1)
        cum_dist = torch.cumsum(r_dist * knn_weight, -1)
        dtm_val = torch.gather(cum_dist + r_dist*(bound-cum_knn_weight), -1, k)
        dtm_val = dtm_val/bound
    else:
        r_dist = knn_dist.pow(r).view(1, 1, HW, -1).expand(batch_size, C, -1, -1)
        cum_dist = torch.cumsum(r_dist * knn_weight, -1)
        dtm_val = torch.gather(cum_dist + r_dist*(bound-cum_knn_weight), -1, k)
        dtm_val = torch.pow(dtm_val/bound, 1/r)
    return dtm_val.squeeze(-1) 


class DTMLayer(nn.Module):
    def __init__(self, m0=0.05, lims=[[1,28], [1,28]], size=[28, 28], r=2):
        """
        Args:
            m0: 
            r: 
            lims:
            size: 
            r:
        """
        super().__init__()
        grid = make_grid(lims, size)    # shape: [(H*W), 2]
        self.dist = cal_dist(grid)
        self.m0 = m0
        self.r = r
        self.flatten = nn.Flatten(start_dim=-2)
        
    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, C, H, W]       # grad

        Returns:
            dtm_val: Tensor of shape [batch_size, C, H, W]
        """
        weight = self.flatten(input)                    # shape: [batch_size, C, (H*W)]
        bound = self.m0 * weight.sum(-1, keepdim=True)  # shape: [batch_size, C, 1]
        
        # find max k among k's of each data s.t. sum({Wi: Wi in (k-1)-NN}) < m0*sum({Wi: i=1...n}) <= sum({Wi: Wi in k-NN})
        with torch.no_grad():
            sorted_weight = torch.sort(weight, -1).values   # shape: [batch_size, C, (H*W)]
            sorted_weight_cumsum = sorted_weight.cumsum(-1) # shape: [batch_size, C, (H*W)]
            max_k = torch.searchsorted(sorted_weight_cumsum, bound).max().item() + 1
            if max_k > weight.shape[-1]:    # when max_k is out of range (max_k > H*W)
                max_k = weight.shape[-1]

        self.dist = self.dist.to(weight.device)
        knn_dist, knn_index = self.dist.topk(max_k, largest=False, dim=-1)  # shape: [(H*W), max_k]
        dtm_val = dtm_using_knn(knn_dist, knn_index, weight, bound, self.r) # shape: [batch_size, C, (H*W)]
        # if self.scale_dtm:
        #     dtm_val = dtm_val * (weight.max(dim=-1, keepdim=True).values / dtm_val.max(dim=-1, keepdim=True).values)  # Think about multiplying weight.max
        return dtm_val.view(*input.shape)