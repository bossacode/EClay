import numpy as np
import torch
import torch.nn as nn
import gudhi

# for [1, H, W] image: lims = [[1, -1], [-1, 1]], size = (H, W)
# for [C, H, W] image: lims = [[-1, 1], [1, -1], [-1, 1]], size = (C, H, W)
def grid_by(lims=[[1,-1], [-1,1]], size=(28, 28)):
    """
    Creates a tensor of grid points.
    Grid points have one-to-one correspondence with input values that are flattened in row-major order.
    
    * D = 2 if 1-channel or D=3 if 3-channel

    Args:
        lims: [domain for C, domain for H, domain for W] if C > 1 or [domain for H, domain for W] if 1-channel
        size: (C, H, W) if C > 1 or (H, W) if 1-channel
    Returns:
        grid: Tensor of shape [(C*H*W), D]
        grid_size: (C, H, W) if C > 1 or (H, W) if 1-channel
    """
    assert len(size) in (2,3) and len(lims) == len(size)
    expansions = [torch.linspace(start, end, steps) for (start, end), steps in zip(lims, size)]
    grid = torch.index_select(torch.cartesian_prod(*expansions),
                              dim=1,
                              index=torch.tensor([0,2,1]) if len(size)==3 else torch.tensor([1,0]))
    grid_size = size
    return grid, grid_size


def knn(X, Y, k, r=2):
    """
    Brute Force KNN.

    Args:
        X: Tensor of shape [batch_size, (C*H*W), D]
        Y: Tensor of shape [(C*H*W), D]
        k: Int representing number of neighbors
        
        * D=2 if 1-channel or D=3 if 3-channel

    Returns:
        dist: Tensor of shape [batch_size, (C*H*W), k]
        index: Tensor of shape [batch_size, (C*H*W), k]
    """
    assert X.shape[1:] == Y.shape
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
    neg_dist = neg_dist.mT                      # shape: [batch_size, (C*H*W), (C*H*W)]
    dist, index = neg_dist.topk(k, dim=-1)
    return -dist, index


def dtm_using_knn(knn_dist, knn_index, weight, weight_bound, r=2):
    """
    Weighted DTM using KNN.

    Args:
        knn_dist: Tensor of shape [batch_size, (C*H*W), k]
        knn_index: Tensor of shape [batch_size, (C*H*W), k]
        weight: Tensor of shape [batch_size, (C*H*W)]
        weight_bound: Tensor of shape [batch_size, 1]
        r: Int r-Norm

    Returns:
        dtm_val: Tensor of shape [batch_size, (C*H*W)]
    """
    CHW = weight.shape[-1]
    weight_bound = weight_bound.unsqueeze(-1)               # shape: [batch_size, 1, 1]
    weight_temp = weight.unsqueeze(1).expand(-1, CHW, -1)   # shape: [batch_size, (C*H*W), (C*H*W)]
    knn_weight = torch.gather(weight_temp, -1, knn_index)   # shape: [batch_size, (C*H*W), k]    

    # finding k's s.t. sum(Xi: Xi in (k-1)-NN) < m0*sum(Xi: i=1...n) <= sum(Xi: Xi in k-NN)
    cum_knn_weight = knn_weight.cumsum(-1)                                                      # shape: [batch_size, (C*H*W), k]
    k_index = torch.searchsorted(cum_knn_weight, weight_bound.repeat(1, CHW, 1))                # shape: [batch_size, (C*H*W), 1]

    if r == 2:
        r_dist = knn_dist.square()
        cum_dist = torch.cumsum(r_dist * knn_weight, -1)                                        # shape: [batch_size, (C*H*W), k]
        dtm_val = torch.gather(cum_dist + r_dist*(weight_bound-cum_knn_weight), -1, k_index)    # shape: [batch_size, (C*H*W), 1]
        dtm_val = torch.sqrt(dtm_val/weight_bound)
    elif r == 1:
        r_dist = knn_dist
        cum_dist = torch.cumsum(r_dist * knn_weight, -1)
        dtm_val = torch.gather(cum_dist + r_dist*(weight_bound-cum_knn_weight), -1, k_index)
        dtm_val = dtm_val/weight_bound
    else:
        r_dist = knn_dist.pow(r)
        cum_dist = torch.cumsum(r_dist * knn_weight, -1)
        dtm_val = torch.gather(cum_dist + r_dist*(weight_bound-cum_knn_weight), -1, k_index)
        dtm_val = torch.pow(dtm_val/weight_bound, 1/r)
    return dtm_val.squeeze(-1) 


class DTMLayer(nn.Module):
    def __init__(self, m0=0.3, lims=[[1,-1], [-1,1]], size=(28, 28), r=2):
        super().__init__()
        self.m0 = m0
        self.r = r
        self.grid, self.grid_size = grid_by(lims, size)

    def dtm(self, input, weight):
        """
        Weighted DTM using KNN.

        Args:
            input: Tensor of shape [batch_size, (C*H*W), D]
            weight: Tensor of shape [batch_size, (C*H*W)]

            * D=2 if input image is 1-channel and D=3 if 3-channel

        Returns:
            dtm_val: Tensor of shape [batch_size, (C*H*W)]
            knn_index: Tensor of shape [batch_size, (C*H*W), k]
            weight_bound: Tensor of shape [batch_size, 1]
        """
        weight_bound = self.m0 * weight.sum(-1, keepdim=True)               # [batch_size, 1]
        
        # finding max k among k's s.t. sum(Xi: Xi in (k-1)-NN) < m0*sum(Xi: i=1...n) <= sum(Xi: Xi in k-NN)
        with torch.no_grad():
            sorted_weight = torch.sort(weight, -1).values                   # [batch_size, (C*H*W)]
            sorted_weight_cumsum = sorted_weight.cumsum(-1)                 # [batch_size, (C*H*W)]
            index = torch.searchsorted(sorted_weight_cumsum, weight_bound)  # [batch_size, 1]
            max_k = index.max().item() + 1

        knn_distance, knn_index = knn(input, self.grid, max_k)

        ##################################################################
        # return 값들 추후에 쓸모없으면 수정
        ##################################################################
        return dtm_using_knn(knn_distance, knn_index, weight, weight_bound, self.r), knn_index, weight_bound

    def forward(self, input, weight):
        """
        Args:
            inputs: Tensor of shape [batch_size, (C*H*W), D]
            weight: Tensor of shape [batch_size, (C*H*W)]

        Returns:
            outputs: Tensor of shape [batch_size, (C*H*W)]
        """
        dtm_val, knn_index, weight_bound = self.dtm(input, weight)
        return dtm_val


class PersistenceLandscapeCustomGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tseq=[0.5, 0.7, 0.9], K_max=2, grid_size=[28, 28], dimensions=[0, 1]):
        """
        Args:
            input: Tensor of shape [batch_size, (C*H*W)]
        Returns:
            landscape: Tensor of shape [batch_size, len_dim, len_tseq, k_max]
            gradient: Tensor of shape [batch_size, len_dim, len_tseq, k_max, (C*H*W)]
        """
        tseq = np.array(tseq)
        land_list = []
        diff_list = []
        ###############################################################
        # for loop over batch (chech if parallelizable)
        ###############################################################
        for n_batch in range(input.shape[0]):
            dtm_val = input[n_batch].numpy()
            cub_cpx = gudhi.CubicalComplex(dimensions=grid_size, top_dimensional_cells=dtm_val)
            ph = cub_cpx.persistence(homology_coeff_field=2, min_persistence=0)      # list of pairs(dimension, (birth, death))
            # 이거 문서 읽으면서 다시 봐보기
            location = cub_cpx.cofaces_of_persistence_pairs()                        # list of 2 lists of numpy arrays with index correspoding to (birth, death)

            if location[0]:
                location_vstack = [np.vstack(location[0]), np.vstack(location[1])]
            else:
                location_vstack = [np.zeros((0,2), dtype=np.int64), np.vstack(location[1])]

            birth_location = np.concatenate((location_vstack[0][:, 0], location_vstack[1][:, 0]))
            death_location = location_vstack[0][:, 1]

            # lengths
            len_dim = len(dimensions)
            len_tseq = len(tseq)
            len_ph = len(ph)

            land = np.zeros((len_dim, len_tseq, K_max))
            land_diff_birth = np.zeros((len_dim, len_tseq, K_max, len_ph))
            land_diff_death = np.zeros((len_dim, len_tseq, K_max, len_ph))

            for i_dim, dim in enumerate(dimensions):
                # select "dim" dimensional persistent homologies
                dim_ph = [pair for pair in ph if pair[0] == dim]
                dim_ph_id = np.array([j for j, pair in enumerate(ph) if pair[0] == dim])

                # number of "dim" dimensional persistent homologies
                len_dim_ph = len(dim_ph)

                # calculate persistence landscapes
                fab = np.zeros((len_tseq, max(len_dim_ph, K_max)))
                for p in range(len_dim_ph):
                    for t in range(len_tseq):
                        fab[t, p] = max(min(tseq[t]-dim_ph[p][1][0], dim_ph[p][1][1]-tseq[t]), 0)
                land[i_dim] = -np.sort(-fab, axis=-1)[:, :K_max]
                land_ind = np.argsort(-fab, axis=-1)[:, :K_max]    # shape: [len_tseq, k_max]

                # derivative
                fab_diff_birth = np.zeros((len_tseq, len_dim_ph))
                for p in range(len_dim_ph):
                    # (t > birth) & (t < (birth + death)/2)
                    fab_diff_birth[:, p] = np.where((tseq > dim_ph[p][1][0]) & (2*tseq < dim_ph[p][1][0] + dim_ph[p][1][1]),
                                                    -1.,
                                                    0.)
                fab_diff_death = np.zeros((len_tseq, len_dim_ph))
                for p in range(len_dim_ph):
                    # (t < death) & (t > (birth + death)/2)
                    fab_diff_death[:, p] = np.where((tseq < dim_ph[p][1][1]) & (2*tseq > dim_ph[p][1][0] + dim_ph[p][1][1]),
                                                    1.,
                                                    0.)
                # derivative of landscape functions with regard to persistence diagram
                for p in range(len_dim_ph):
                    land_diff_birth[i_dim, :, :, dim_ph_id[p]] = np.where(p == land_ind,
                                                                    np.repeat(np.expand_dims(fab_diff_birth[:, p], -1), K_max, -1),
                                                                    0)
                for p in range(len_dim_ph):
                    land_diff_death[i_dim, :, :, dim_ph_id[p]] = np.where(p == land_ind,
                                                                    np.repeat(np.expand_dims(fab_diff_death[:, p], -1), K_max, -1),
                                                                    0)
            land_list.append(land)
            
            # derivative of persistence diagram with regard to input: dDx/dX
            DiagFUNDiffBirth = np.zeros((len_ph, len(dtm_val)))
            for iBirth in range(len(birth_location)):
                DiagFUNDiffBirth[iBirth, birth_location[iBirth]] = 1

            DiagFUNDiffDeath = np.zeros((len_ph, len(dtm_val)))
            for iDeath in range(len(death_location)):
                DiagFUNDiffDeath[iDeath, death_location[iDeath]] = 1	

            if location[0]:
                dimension = np.concatenate((np.hstack([np.repeat(ldim, len(location[0][ldim])) for ldim in range(len(location[0]))]),
                                            np.hstack([np.repeat(ldim, len(location[1][ldim])) for ldim in range(len(location[1]))])))
            else:
                dimension = np.hstack([np.repeat(ldim, len(location[1][ldim])) for ldim in range(len(location[1]))])
            if len(death_location) > 0:
                persistence = np.concatenate((dtm_val[death_location], np.repeat(np.infty, len(np.vstack(location[1]))))) - dtm_val[birth_location]
            else:
                persistence = np.repeat(np.infty, len(np.vstack(location[1])))
            order = np.lexsort((-persistence, -dimension))

            diff = np.dot(land_diff_birth, DiagFUNDiffBirth[order, :]) + np.dot(land_diff_death, DiagFUNDiffDeath[order, :])
            diff_list.append(diff)

        landscape = torch.from_numpy(np.stack(land_list)).to(torch.float32)
        gradient = torch.from_numpy(np.stack(diff_list)).to(torch.float32)
        ####################################################################
        print("local_grad shape:", gradient.shape)
        ####################################################################
        ctx.save_for_backward(gradient)
        return landscape, gradient

    @staticmethod
    def backward(ctx, grad_out, _grad_out_gradient):
        local_grad = ctx.saved_tensors
        ###################################################################################
        print("grad_out shape:", grad_out.shape)
        print("_grad_out_gradient shape:", _grad_out_gradient.shape)
        print("all 0:", torch.all(_grad_out_gradient == 0.))
        ###################################################################################
        grad_input = torch.einsum('...ijk,...ijkl->...l', grad_out, local_grad)
        # gradient에 대한 gradient 누적해야 하나...?
        print(_grad_out_gradient)   # 요거 0이면 누적 안 해도 될텐데
        return grad_input, None, None, None, None


class PersistenceLandscapeLayer(nn.Module):
    def __init__(self, tseq=[0.5, 0.7, 0.9], K_max=2, grid_size=[28, 28], dimensions=[0, 1]):
        super().__init__()
        self.tseq = np.array(tseq)
        self.K_max = K_max
        self.grid_size = grid_size
        self.dimensions = dimensions

    def forward(self, inputs):
        """
        Args:
            input: Tensor of shape [batch_size, (C*H*W)]
        Returns:
            landscape: Tensor of shape [batch_size, len_dim, len_tseq, k_max]
        """
        return PersistenceLandscapeCustomGrad.apply(inputs, self.tseq, self.K_max, self.grid_size, self.dimensions)[0]


class WeightedAvgLandscapeLayer(nn.Module):
    def __init__(self, K_max=2, dimensions=[0, 1]):
        super().__init__()
        self.land_weight = nn.Parameter(torch.tensor(1/K_max).repeat(1, len(dimensions), 1, K_max)) # weight of landscapes initialized as uniform
        self.softmax = nn.Softmax(dim=-1)
        self.flatten = nn.Flatten()

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, len_dim, len_tseq, k_max]

        Returns:
            output: Tensor of shape [batch_size, (len_dim*len_tseq)]
        """
        weight = self.softmax(self.land_weight)
        weighted_avg_land = torch.sum(input * weight, dim=-1)   # weighted average of landscapes
        output = self.flatten(weighted_avg_land)
        return output


class GThetaLayer(nn.Module):
    def __init__(self, out_features, tseq=[0.5, 0.7, 0.9], dimensions=[0, 1]):
        super().__init__()
        self.g_layer = nn.Linear(len(dimensions)*len(tseq), out_features)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, (len_dim*len_tseq)]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        output = self.g_layer(input)
        return output


class TopoWeightLayer(nn.Module):
    def __init__(self, out_features, m0=0.3, lims=[[1,-1], [-1,1]], size=(28, 28), r=2, tseq=[0.5, 0.7, 0.9], K_max=2, dimensions=[0, 1]):
        super().__init__()
        self.dtm_layer = DTMLayer(m0, lims, size, r)
        self.landscape_layer = PersistenceLandscapeLayer(tseq, K_max, self.dtm_layer.grid_size, dimensions)
        self.avg_layer = WeightedAvgLandscapeLayer(K_max, dimensions)
        self.gtheta_layer = GThetaLayer(out_features, tseq, dimensions)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, (C*H*W)]

        Returns:
            output: Tensor of shape [batch_size, out_features]
        """
        grids = self.dtm_layer.grid.expand(input.shape[0],-1, -1)
        dtm_val = self.dtm_layer(input=grids, weight=input)
        land = self.landscape_layer(dtm_val)
        weighted_avg_land = self.avg_layer(land)
        output = self.gtheta_layer(weighted_avg_land)
        return output