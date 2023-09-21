import numpy as np
import torch
import torch.nn as nn
import gudhi
import time

# for [1, 28, 28] image: lims = [[1, -1], [-1,1]], by= [-1/13.5, 1/13.5]
# for [3, 28, 28] image: lims = [[-1, 1], [1, -1], [-1, 1]], by= [1, -1/13.5, 1/13.5]
def grid_by(lims=[[1,-1], [-1,1]], by=[-1/13.5, 1/13.5]):
    """
    Creates a tensor of grid points with shape [(C*H*W), D]
    Grid points have one-to-one correspondnce with pixel values flattened in row-major order
    
    * D=2 if input image is 1-channel and D=3 if 3-channel

    Args:
        lims: domain of the grid points
        by: interval between each grid point
    """
    expansions = [torch.arange(start, end+step, step, dtype=torch.float32) for (start, end), step in zip(lims, by)]
    grid_size = [len(ex) for ex in expansions]  # [H, W] if 1-channel or [C, H, W] if 3-channel
    grid = torch.index_select(torch.cartesian_prod(*expansions),
                              dim=1,
                              index=torch.tensor([0,2,1]) if len(lims)==3 else torch.tensor([1,0]))
    return grid, grid_size


def knn(X, Y, k, r=2):
    """
    Brute Force KNN.

    Args:
        X: Tensor of shape [batch_size, (C*H*W), D]
        Y: Tensor of shape [(C*H*W), D]
        k: Int representing number of neighbors
        
        * D=2 if input image is 1-channel and D=3 if 3-channel

    Returns:
        distance: Tensor of shape [batch_size, (C*H*W), k]
        index: Tensor of shape [batch_size, (C*H*W), k]
    """
    # print(X.shape, Y.shape)
    assert X.shape[-1] == Y.shape[1]
    d = X.shape[-1]
    if r == 2:
        Xr = X.unsqueeze(2)
        Yr = Y.view(1, 1, -1, d)
        neg_dist = -torch.sqrt(torch.sum((Xr - Yr)**2, -1))
    elif r == 1:
        Xr = X.unsqueeze(2)
        Yr = Y.view(1, 1, -1, d)
        neg_dist = -torch.sum(torch.abs(Xr - Yr), -1)
    else:
        Xr = X.unsqueeze(2)
        Yr = Y.view(1, 1, -1, d)
        neg_dist = -torch.pow(torch.sum((Xr - Yr)**r, -1), 1/r)
    neg_dist = neg_dist.mT                          # neg_dist shape: [batch_size, (C*H*W), (C*H*H)]
    distance, index = neg_dist.topk(k, dim=-1)      # distance shape: [batch_size, (C*H*W), k]
    return -distance, index


def dtm_using_knn(knn_distance, knn_index, weight, weight_bound, r=2):
    """
    Weighted Distance to measure using KNN.

    Args:
        knn_distance: Tensor of shape [batch_size, (C*W*H), k]
        knn_index: Tensor of shape [batch_size, (C*W*H), k]
        weight: Tensor of shape [batch_size, (C*W*H)]
        weight_bound: Tensor of shape [batch_size, 1]
        r: Int r-Norm

    Returns:
        dtm_val: Tensor of shape [batch_size, (C*W*H)]
    """
    size = weight.shape[-1]
    weight_bound = weight_bound.unsqueeze(-1)               # [batch_size, 1, 1]
    weight_temp = weight.unsqueeze(1).expand(-1, size, -1)  # [batch_size, (C*H*W), (C*H*W)]
    knn_weight = torch.gather(weight_temp, -1, knn_index)   # [batch_size, (C*H*W), k]    

    # finding indexes of k s.t. sum(Xi: Xi in (k-1)-NN) < m0*sum(Xi: i=1...n) <= sum(Xi: Xi in k-NN)
    with torch.no_grad():
        cum_knn_weight = knn_weight.cumsum(-1)                                                   # [batch_size, (C*H*W), k]
        index = torch.searchsorted(cum_knn_weight, weight_bound.repeat(1, size, 1))              # [batch_size, (C*H*W), 1]

    if r == 2:
        dist_temp = knn_distance.square()
        cum_dist = torch.cumsum(dist_temp * knn_weight, -1)                                         # [batch_size, (C*H*W), k]
        dtm_val = torch.gather(cum_dist + dist_temp*(weight_bound-cum_knn_weight), -1, index)    # [batch_size, (C*H*W), 1]
        dtm_val = torch.sqrt(dtm_val/weight_bound)
    elif r == 1:
        dist_temp = knn_distance
        cum_dist = torch.cumsum(dist_temp * knn_weight, -1)                                         # [batch_size, (C*H*W), k]
        dtm_val = torch.gather(cum_dist + dist_temp*(weight_bound-cum_knn_weight), -1, index)    # [batch_size, (C*H*W), 1]
        dtm_val = dtm_val/weight_bound
    else:
        dist_temp = knn_distance.pow(r)
        cum_dist = torch.cumsum(dist_temp * knn_weight, -1)                                         # [batch_size, (C*H*W), k]
        dtm_val = torch.gather(cum_dist + dist_temp*(weight_bound-cum_knn_weight), -1, index)    # [batch_size, (C*H*W), 1]
        dtm_val = torch.pow(dtm_val/weight_bound, 1/r)
    return dtm_val.squeeze() 


class DTMWeightLayer(nn.Module):
    def __init__(self, m0=0.3, lims=[[1,-1], [-1,1]], by=[-1/13.5, 1/13.5], r=2, **kwargs):
        super().__init__()
        self.m0 = m0
        self.r = r
        self.grid, self.grid_size = grid_by(lims, by)

    def dtm(self, inputs, weight):
        """
        Weighted Distance to measure using KNN.

        Args:
            inputs: Tensor of shape [batch_size, (C*H*W), D]
            weight: Tensor of shape [batch_size, (C*H*W)]

            * D=2 if input image is 1-channel and D=3 if 3-channel

        Returns:
            dtmValue: Tensor of shape [batch_size, (C*H*W)]
            knnIndex: Tensor of shape [batch_size, (C*H*W), k]
            weightBound: Tensor of shape [batch_size, 1]
        """
        weight_bound = self.m0 * weight.sum(-1, keepdim=True)               # [batch_size, 1]
        
        # finding value of k for knn
        with torch.no_grad():
            sorted_weight = torch.sort(weight, -1).values                   # [batch_size, (C*H*W)]
            sorted_weight_cumsum = sorted_weight.cumsum(-1)                 # [batch_size, (C*H*W)]
            index = torch.searchsorted(sorted_weight_cumsum, weight_bound)  # [batch_size, 1]
            max_index = index.max().item() + 1

        knn_distance, knn_index = knn(inputs, self.grid, max_index)
        return dtm_using_knn(knn_distance, knn_index, weight, weight_bound, self.r), knn_index, weight_bound

    def forward(self, inputs, weight):
        """
        Args:
            inputs: Tensor of shape [batch_size, (C*H*W), D]
            weight: Tensor of shape [batch_size, (C*H*W)]

        Returns:
            outputs: Tensor of shape [batch_size, (C*H*W)]
        """
        dtm_val, knn_index, weight_bound = self.dtm(inputs, weight)
        return dtm_val


class PersistenceDiagramLayer(nn.Module):
    def __init__(self, grid_size=[28, 28], dimensions=[0, 1], nmax_diag=100, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.dimensions = dimensions
        self.nmax_diag = nmax_diag

    def python_op_diag(self, fun_value):
        """
        Python domain function to compute landscape.
    
        It also computes things needed for gradient, as we don't want to enter
        python multiple times.

        Args:
            fun_value: numpy array of shape [N]
            -> Change to tensor of shape [(C*H*W)]?

        Returns:
            diagram: numpy array of shape [len(self.dimensions), self.nmax_diag, 2]
        """
        # detached from graph
        # should it share storage?
        fun_value = fun_value.view(self.grid_size).detach()     # [C, H, W]
        cub_cpx = gudhi.CubicalComplex(top_dimensional_cells=fun_value)
        perst = cub_cpx.persistence(homology_coeff_field=2, min_persistence=0)   # list of pairs(dimension, (birth, death))

        perst_list = [None] * len(self.dimensions)
        for i, dim in enumerate(self.dimensions):
            dim_perst = [pair[1] for pair in perst if pair[0] == dim]
            perst_list[i] = dim_perst

        diagram = np.zeros((len(self.dimensions), self.nmax_diag, 2), dtype='float32')
        for i in range(len(self.dimensions)):
            ndim_diag = min(len(perst_list[i], self.nmax_diag))
            if (ndim_diag > 0):
                diagram[i][0:ndim_diag] = perst_list[i][0:ndim_diag]      # torch에 맞춰서 수정 필요 or diagram을 np.zeros로 만들기
        return diagram

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [batch_size, (C*W*H)]

        Returns:
            outputs: Tensor of shape [batch_size, len(self.dimensions), self.nmax_diag, 2]
        """
        diag_list = []
        for i in range(inputs[0]):
            diag_list[i] = self.python_op_diag(inputs[i])   # list of numpy arrays
        return torch.from_numpy(np.stack(diag_list))


class PersistenceLandscapeLayer(nn.Module):
    def __init__(self, tseq=[0.5, 0.7, 0.9], KK=[0, 1], grid_size=[28, 28], dimensions=[0, 1], dtype='float32', **kwargs):
        super().__init__()
        self.dtype = dtype
        self.tseq = np.array(tseq, dtype=dtype)
        self.KK = np.array(KK)
        self.grid_size = grid_size
        self.dimensions = dimensions

    def python_op_diag_landscape(self, fun_value):
        """
        Python domain function to compute landscape.

        It also computes things needed for gradient, as we don't want to enter
        python multiple times.

        Args:
            fun_value: numpy array of shape [N]
            -> Change to tensor of shape [(C*H*W)]?

        Returns:
            land: numpy array of shape [len(dims), len(tseq), len(KK)]
            diff: numpy array of shape [N, len(dims), len(tseq), len(KK)]
        """
        # detach from computational graph
        cub_cpx = gudhi.CubicalComplex(dimensions=self.grid_size, top_dimensional_cells=fun_value.detach())
        perst = cub_cpx.persistence(homology_coeff_field=2, min_persistence=0)      # list of pairs(dimension, (birth, death))
        location = cub_cpx.cofaces_of_persistence_pairs()                           # list of [np.array] with index of fun_value correspoding to (birth, death)

        if location[0]:
            location_vstack = [np.vstack(location[0]), np.vstack(location[1])]
        else:
            location_vstack = [np.zeros((0,2), dtype=np.int64), np.vstack(location[1])]

        birth_location = np.concatenate((location_vstack[0][:, 0], location_vstack[1][:, 0]))
        death_location = location_vstack[0][:, 1]

        # lengths
        len_dim = len(self.dimensions)
        len_tseq = len(self.tseq)
        len_KK = len(self.KK)
        len_perst = len(perst)

        land = np.zeros((len_dim, len_tseq, len_KK), dtype=self.dtype)
        land_diff_birth = np.zeros((len_dim, len_tseq, len_KK, len_perst), dtype=self.dtype)
        land_diff_death = np.zeros((len_dim, len_tseq, len_KK, len_perst), dtype=self.dtype)

        for i, dim in enumerate(self.dimensions):
            # select 0 dimension feature
            dim_perst = [pair for pair in perst if pair[0] == dim]
            dim_perst_id = np.array([j for j, pair in enumerate(perst) if pair[0] == dim])

            # local lengths
            len_dim_perst = len(dim_perst)

            # arrange
            