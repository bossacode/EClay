import torch
import torch.nn as nn


def make_grid(lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28]):
    """Creates a tensor of 2D grid points. Grid points have one-to-one correspondence with image pixels flattened in row-major order.

    Args:
        lims (list, optional): Domain of x & y axis. Defaults to [[-0.5, 0.5], [-0.5, 0.5]].
        size (list, optional): Number of discretized points for x & y axis. Defaults to [28, 28].

    Returns:
        grid (torch.Tensor): Grid coordinates. Tensor of shape [(H*W), 2].
    """
    assert len(size) == 2 and len(lims) == len(size)
    x_seq = torch.linspace(start=lims[0][0], end=lims[0][1], steps=size[0])
    y_seq = torch.linspace(start=lims[1][1], end=lims[1][0], steps=size[1])
    x_coord, y_coord = torch.meshgrid(x_seq, y_seq, indexing="xy")
    grid = torch.concat([x_coord.reshape(-1, 1), y_coord.reshape(-1, 1)], dim=1)
    return grid


def pc2grid_dist(X, grid, r=2):
    """Calculate distance between all points in point cloud and all grid cooridnates.

    Args:
        X (torch.Tensor): Batch of point clouds. Tensor of shape [B, N, D].
        grid (torch.Tensor): Grid coordinates. Tensor of shape [(H*W), D].
        r (int, optional): r-norm. Defaults to 2.

    Returns:
        dist (torch.Tensor): Distance between all points and grid coordinates. Tensor of shape [B, (H*W), N].
    """
    assert X.shape[-1] == grid.shape[-1]
    X = X.unsqueeze(-2)                 # shape: [B, N, 1, D]
    Y = grid.view(1, 1, *grid.shape)    # shape: [1, 1, (H*W), D]
    if r == 2:
        dist = torch.sqrt(torch.sum((X - Y)**2, dim=-1))    # shape: [B, N, (H*W)]
    elif r == 1:
        dist = torch.sum(torch.abs(X - Y), dim=-1)
    else:
        dist = torch.pow(torch.sum((X - Y)**r, dim=-1), 1/r)
    return dist.transpose(1, 2)


def grid2grid_dist(grid, r=2):
    """Calculate distance between all grid cooridnates.

    Args:
        grid (torch.Tensor): Grid coordinates. Tensor of shape [(H*W), D].
        r (int, optional): r-norm. Defaults to 2.
    
    Returns:
        dist (torch.Tensor): Distance between all grid coordinates. Tensor of shape [(H*W), (H*W)].
    """
    X = grid.unsqueeze(1)   # shape: [(H*W), 1, D]
    Y = grid.unsqueeze(0)   # shape: [1, (H*W), D]
    if r == 2:
        dist = torch.sqrt(torch.sum((X - Y)**2, dim=-1))    # shape: [(H*W), (H*W)]
    elif r == 1:
        dist = torch.sum(torch.abs(X - Y), dim=-1)
    else:
        dist = torch.pow(torch.sum((X - Y)**r, dim=-1), 1/r)
    return dist


def dtm_using_knn(knn_dist, bound, r=2):
    """DTM using KNN.

    Args:
        knn_dist (torch.Tensor): Distance to k-nearest points for each grid coordinate. Tensor of shape [B, (H*W), k].
        bound (float): Weight bound that corresponds to m0*sum({Wi: i=1...n}).
        r (int, optional): r-norm. Defaults to 2.

    Returns:
        dtm_val (torch.Tensor): DTM value. Tensor of shape [B, (H*W)].
    """
    cum_knn_weight = torch.math.ceil(bound)
    if r == 2:
        r_dist = knn_dist.square()
        cum_dist = torch.cumsum(r_dist, dim=-1)                                     # shape: [B, (H*W), k]
        dtm_val = cum_dist[..., -1] + r_dist[..., -1]*(bound - cum_knn_weight)      # shape: [B, (H*W)]
        dtm_val = torch.sqrt(dtm_val/bound)
    elif r == 1:
        r_dist = knn_dist
        cum_dist = torch.cumsum(r_dist, dim=-1)
        dtm_val = cum_dist[..., -1] + r_dist[..., -1]*(bound - cum_knn_weight)
        dtm_val = dtm_val/bound
    else:
        r_dist = knn_dist.pow(r)
        cum_dist = torch.cumsum(r_dist, dim=-1)
        dtm_val = cum_dist[..., -1] + r_dist[..., -1]*(bound - cum_knn_weight)
        dtm_val = torch.pow(dtm_val/bound, 1/r)
    return dtm_val


def weighted_dtm_using_knn(knn_dist, knn_index, weight, bound, r=2):
    """Weighted DTM using KNN.

    Args:
        knn_dist (torch.Tensor): Distance to max_k-nearest points for each grid coordinate. Tensor of shape [(H*W), max_k].
        knn_index (torch.Tensor): Index of max_k-nearest points in knn_dist. Tensor of shape [(H*W), max_k].
        weight (torch.Tensor): Weight used for DTM. Tensor of shape [B, C, (H*W)].
        bound (torch.Tensor): Weight bound that corresponds to m0*sum({Wi: i=1...n}) for each data. Tensor of shape [B, C, 1].
        r (int, optional): r-norm. Defaults to 2.

    Returns:
        dtm_val (torch.Tensor): DTM value. Tensor of shape [B, C, (H*W)].
    """
    B, C, HW = weight.shape
    
    weight_temp = weight.unsqueeze(2).expand(-1, -1, HW, -1)            # shape: [B, C, (H*W), (H*W)]
    knn_index = knn_index.view(1, 1, HW, -1).expand(B, C, -1, -1)       # shape: [B, C, (H*W), max_k]
    knn_weight = torch.gather(weight_temp, dim=-1, index=knn_index)     # shape: [B, C, (H*W), max_k]    

    # finding k's s.t. sum({Wi: Wi in (k-1)-NN}) < bound <= sum({Wi: Wi in k-NN}) for each data
    cum_knn_weight = torch.cumsum(knn_weight, dim=-1)                   # shape: [B, C, (H*W), max_k]
    bound = bound.unsqueeze(-1)                                         # shape: [B, C, 1, 1]
    k = torch.searchsorted(cum_knn_weight, bound.repeat(1, 1, HW, 1))   # shape: [B, C, (H*W), 1]
    
    # prevent index out of bounds error when some values of k_index equal HW
    if (k == HW).any():
        k[k == HW] -= 1

    if r == 2:
        r_dist = knn_dist.square().view(1, 1, HW, -1).expand(B, C, -1, -1)                  # shape: [B, C, (H*W), max_k]
        cum_dist = torch.cumsum(r_dist * knn_weight, dim=-1)                                # shape: [B, C, (H*W), max_k]
        dtm_val = torch.gather(cum_dist + r_dist*(bound - cum_knn_weight), dim=-1, index=k) # shape: [B, C, (H*W), 1]
        dtm_val = torch.sqrt(dtm_val/bound)
    elif r == 1:
        r_dist = knn_dist.view(1, 1, HW, -1).expand(B, C, -1, -1)
        cum_dist = torch.cumsum(r_dist * knn_weight, dim=-1)
        dtm_val = torch.gather(cum_dist + r_dist*(bound - cum_knn_weight), dim=-1, index=k)
        dtm_val = dtm_val/bound
    else:
        r_dist = knn_dist.pow(r).view(1, 1, HW, -1).expand(B, C, -1, -1)
        cum_dist = torch.cumsum(r_dist * knn_weight, dim=-1)
        dtm_val = torch.gather(cum_dist + r_dist*   (bound - cum_knn_weight), dim=-1, index=k)
        dtm_val = torch.pow(dtm_val/bound, 1/r)
    return dtm_val.squeeze(-1) 


class DTMLayer(nn.Module):
    def __init__(self, m0=0.05, lims=[[0.0125, 0.9875], [0.0125, 0.9875]], size=[40, 40], r=2):
        """
        Args:
            m0 (float, optional): Parameter between 0 and 1 that controls locality. Defaults to 0.05.
            lims (list, optional): Domain of x & y axis. Defaults to [[0.0125, 0.9875], [0.0125, 0.9875]].
            size (list, optional): Number of discretized points for x & y axis. Defaults to [40, 40].
            r (int, optional): r-norm. Defaults to 2.
        """
        super().__init__()
        self.grid = make_grid(lims, size)   # shape: [(H*W), 2]
        self.m0 = m0
        self.size = size
        self.r = r
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Point clouds. Tensor of shape [B, N, D].

        Returns:
            dtm_val (torch.Tensor): DTM value. Tensor of shape [B, H, W].
        """
        bound = self.m0 * x.shape[-2]
        dist = pc2grid_dist(x, self.grid)                               # shape: [B, (H*W), N]
        # knn, find k s.t. k-1 < bound <= k
        k = torch.math.ceil(bound)
        knn_dist, knn_index = dist.topk(k, largest=False, dim=-1)   # shape: [B, (H*W), k]
        # dtm
        dtm_val = dtm_using_knn(knn_dist, bound, self.r)            # shape: [B, (H*W)]
        return dtm_val.view(x.shape[0], *self.size).unsqueeze(1)    # shape: [B, 1, H, W]


class WeightedDTMLayer(nn.Module):
    def __init__(self, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28], r=2):
        """
        Args:
            m0 (float, optional): Parameter between 0 and 1 that controls locality. Defaults to 0.05.
            lims (list, optional): Domain of x & y axis. Defaults to [[-0.5, 0.5], [-0.5, 0.5]].
            size (list, optional): Number of discretized points for x & y axis. Defaults to [28, 28].
            r (int, optional): r-norm. Defaults to 2.
        """
        super().__init__()
        grid = make_grid(lims, size)    # shape: [(H*W), 2]
        self.dist = grid2grid_dist(grid)      # shape: [(H*W), (H*W)]
        self.m0 = m0
        self.r = r
        self.flatten = nn.Flatten(start_dim=-2)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Images. Tensor of shape [B, C, H, W].

        Returns:
            dtm_val (torch.Tensor): DTM value. Tensor of shape [B, C, H, W].
        """
        weight = self.flatten(x)                        # shape: [B, C, (H*W)]
        bound = self.m0 * weight.sum(-1, keepdim=True)  # shape: [B, C, 1]
        self.dist = self.dist.to(weight.device)
        # knn, find max k among k's of each data s.t. sum({Wi: Wi in (k-1)-NN}) < bound <= sum({Wi: Wi in k-NN})
        with torch.no_grad():
            sorted_weight = torch.sort(weight, dim=-1).values   # shape: [B, C, (H*W)]
            sorted_weight_cumsum = sorted_weight.cumsum(dim=-1) # shape: [B, C, (H*W)]
            max_k = torch.searchsorted(sorted_weight_cumsum, bound).max().item() + 1
            if max_k > weight.shape[-1]:    # when max_k is out of range, i.e., max_k > (H*W)
                max_k = weight.shape[-1]
        knn_dist, knn_index = self.dist.topk(max_k, largest=False, dim=-1)  # shape: [(H*W), max_k]
        # dtm
        dtm_val = weighted_dtm_using_knn(knn_dist, knn_index, weight, bound, self.r) # shape: [B, C, (H*W)]
        return dtm_val.view(*x.shape)