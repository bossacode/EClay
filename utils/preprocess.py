import torch
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from utils.dtm import WeightedDTMLayer, DTMLayer


# add corruption and noise to MNIST data
def corrupt_noise(x, p):
    """Replace pixels with random noise between 0 and 1 with probabilty p.

    Args:
        x (torch.Tensor): Original image data of shape [B, C, H, W]
        p (float): Corruption and noise probability.

    Returns:
        torch.Tensor: Data corrupted by random noise.
    """
    data_list = []
    distr = Bernoulli(probs=p)
    for data in x:
        data_crpt = torch.where(distr.sample(data.shape).bool(),
            0,
            data)
        data_crpt_noise = torch.where((data < 0.01) * distr.sample(data.shape).bool(),
            torch.rand(data.shape),
            data_crpt)
        data_list.append(data_crpt_noise)
    x_crpt_noise = torch.stack(data_list, dim=0)
    return x_crpt_noise


# add noise to ORBIT data
def noise(x, p):
    """Replace points with random noise between 0 and 1 with probabilty p.

    Args:
        x (torch.Tensor): One point cloud from original data. Shape [B, num_pts, D]
        p (float): Noise probability.

    Returns:
        torch.Tensor: Data corrupted by random noise.
    """
    x_noise = x.clone()
    distr = Bernoulli(probs=p)
    for i in range(len(x)):
        noise_ind, = distr.sample(sample_shape=(x.shape[1],)).nonzero(as_tuple=True)
        x_noise[i, noise_ind, :] = torch.rand(len(noise_ind), x.shape[-1])
    return x_noise


def dtm_transform(x, m0, lims, size, weighted=True):
    """Tranform data using DTM.

    Args:
        x (torch.Tensor): Tensor of shape [B, C, H, W].
        weighted (bool): Use weighted DTM if True.

    Returns:
        torch.Tensor: Tensor of shape [B, C, H, W].
    """
    if weighted:
        dtm = WeightedDTMLayer(m0, lims, size)
    else:
        dtm = DTMLayer(m0, lims, size)
    
    data_list = []
    dataloader = DataLoader(x, batch_size=64)
    for data in dataloader:
        data_list.append(dtm(data))
    return torch.concat(data_list, dim=0)