import torch
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from utils.dtm import WeightedDTMLayer, DTMLayer


def corrupt_noise(x, p):
    """Replace pixels with random noise between 0 and 1 with probabilty p.

    Args:
        x (torch.Tensor): Original data.
        p (float): Corruption and noise probability.

    Returns:
        torch.Tensor: Data corrupted by random noise.
    """
    dist = Bernoulli(probs=p)
    x_crpt = torch.where(dist.sample(x.shape).bool(),
                        0,
                        x)
    x_crpt_noise = torch.where(dist.sample(x.shape).bool(),
                            torch.rand(x.shape),
                            x_crpt)
    return x_crpt_noise


def cn_transform(x, p):
    data_list = []
    dataloader = DataLoader(x, batch_size=64)
    for data in dataloader:
        data_list.append(corrupt_noise(data, p))
    return torch.concat(data_list, dim=0)


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