import torch
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from dtm import DTMLayer
from ec import CubECC2d
from pl import CubPL2d
import numpy as np
import os


def add_noise(x, p):
    """
    Replace pixels with random noise between 0 and 1 with probabilty p.

    Args:
        x (torch.Tensor): Original data.
        p (float): Noise probability.

    Returns:
        torch.Tensor: Data corrupted by random noise.
    """
    dist = Bernoulli(probs=p)
    x_noise = torch.where(dist.sample(x.shape).bool(),
                          torch.rand(x.shape),
                          x)
    return x_noise


def dtm_transform(x, dtm):
    """
    Tranform data using DTM.

    Args:
        x (torch.Tensor): Tensor of shape [B, C, H, W].
        dtm (dtm.DTMLayer): Instance of DTMLayer class.

    Returns:
        torch.Tensor: Tensor of shape [B, C, H, W].
    """
    data_list = []
    dataloader = DataLoader(x, batch_size=32)
    for data in dataloader:
        data_list.append(dtm(data))
    return torch.concat(data_list, dim=0)


def ecc_transform(x, ecc):
    """
    Transform data to Euler Characteristic Curve.

    Args:
        x (torch.Tensor): Tensor of shape [B, C, H, W].
        ecc (ec.CubECC2d): Instance of CubECC2d class.

    Returns:
        torch.Tensor: Tensor of shape [B, C, T].
    """
    data_list = []
    dataloader = DataLoader(x, batch_size=32)
    for data in dataloader:
        data_list.append(ecc(data))
    return torch.concat(data_list, dim=0)


def pl_transform(x, pl):
    """
    Transform data to Persistent Landscape.

    Args:
        x (torch.Tensor):  Tensor of shape [B, C, H, W].
        pl (pl.CubPL2d): Instance of CubPL2d class.

    Returns:
        torch.Tensor: Tensor of shape [B, C, len_dim, K_max, T].
    """
    data_list = []
    dataloader = DataLoader(x, batch_size=32)
    for data in dataloader:
        data_list.append(pl(data))
    return torch.concat(data_list, dim=0)


def generate_data(n_train, n_val, noise_prob_list):
    """_summary_

    Args:
        n_train (int): Number of training data to sample for each label.
        n_val (int): Number of validation data to sample for each label.
        noise_prob_list (list): List containing noise probability.
    """
    train_data = MNIST(root="./raw_data", train=True, download=True, transform=ToTensor())  # shape: (60000, 28, 28)
    test_data = MNIST(root="./raw_data", train=False, download=True, transform=ToTensor())  # shape: (10000, 28, 28) 
    
    # normalize and add channel dimension: (N, C, H, W)
    x_train = (train_data.data / 255).unsqueeze(1)
    y_train = train_data.targets
    x_test = (test_data.data / 255).unsqueeze(1)
    y_test = test_data.targets

    # sample "n_train" training data and "n_val" validation data for each label
    # total training data size is n_train * # of labels
    train_idx_list = []
    val_idx_list = []
    for label in y_train.unique():
        idx = torch.where(y_train == label)[0]
        sampled_idx = np.random.choice(idx, size=n_train + n_val, replace=False)
        train_idx_list.append(sampled_idx[:n_train])
        val_idx_list.append(sampled_idx[n_train:])
    train_idx = np.concatenate(train_idx_list)
    val_idx = np.concatenate(val_idx_list)
    x_train_sampled, y_train_sampled = x_train[train_idx], y_train[train_idx]
    x_val_sampled, y_val_sampled = x_train[val_idx], y_train[val_idx]

    dtm005 = DTMLayer(m0=0.05, lims=[[1, 28], [1, 28]], size=[28, 28])
    ecc_dtm005 = CubECC2d(as_vertices=False, sublevel=True, size=[28, 28], interval=[0, 7, 32])
    pl_dtm005 = CubPL2d(superlevel=False, tseq=[0, 7, 32], K_max=2, dimensions=[0, 1])

    dtm02 = DTMLayer(m0=0.2, lims=[[1, 28], [1, 28]], size=[28, 28])
    ecc_dtm02 = CubECC2d(as_vertices=False, sublevel=True, size=[28, 28], interval=[1, 8, 32])
    pl_dtm02 = CubPL2d(superlevel=False, tseq=[1, 8, 32], K_max=3, dimensions=[0, 1])

    for p in noise_prob_list:
        dir_name = f"generated_data/data_{n_train*10}/noise_" + str(int(p * 100)).zfill(2) + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # train data
        x_train_noise = add_noise(x_train_sampled, p)
        x_train_dtm005 = dtm_transform(x_train_noise, dtm005)
        ecc_train_dtm005 = ecc_transform(x_train_dtm005, ecc_dtm005)
        pl_train_dtm005 = pl_transform(x_train_dtm005, pl_dtm005)
        x_train_dtm02 = dtm_transform(x_train_noise, dtm02)
        ecc_train_dtm02 = ecc_transform(x_train_dtm02, ecc_dtm02)
        pl_train_dtm02 = pl_transform(x_train_dtm02, pl_dtm02)
        
        # validation data
        x_val_noise = add_noise(x_val_sampled, p)
        x_val_dtm005 = dtm_transform(x_val_noise, dtm005)
        ecc_val_dtm005 = ecc_transform(x_val_dtm005, ecc_dtm005)
        pl_val_dtm005 = pl_transform(x_val_dtm005, pl_dtm005)
        x_val_dtm02 = dtm_transform(x_val_noise, dtm02)
        ecc_val_dtm02 = ecc_transform(x_val_dtm02, ecc_dtm02)
        pl_val_dtm02 = pl_transform(x_val_dtm02, pl_dtm02)

        # test data
        x_test_noise = add_noise(x_test, p)
        x_test_dtm005 = dtm_transform(x_test_noise, dtm005)
        ecc_test_dtm005 = ecc_transform(x_test_dtm005, ecc_dtm005)
        pl_test_dtm005 = pl_transform(x_test_dtm005, pl_dtm005)
        x_test_dtm02 = dtm_transform(x_test_noise, dtm02)
        ecc_test_dtm02 = ecc_transform(x_test_dtm02, ecc_dtm02)
        pl_test_dtm02 = pl_transform(x_test_dtm02, pl_dtm02)

        torch.save((x_train_noise, ecc_train_dtm005, ecc_train_dtm02, pl_train_dtm005, pl_train_dtm02, y_train_sampled), dir_name + "train.pt")
        torch.save((x_val_noise, ecc_val_dtm005, ecc_val_dtm02, pl_val_dtm005, pl_val_dtm02, y_val_sampled), dir_name + "val.pt")
        torch.save((x_test_noise, ecc_test_dtm005, ecc_test_dtm02, pl_test_dtm005, pl_test_dtm02, y_test), dir_name + "test.pt")


if __name__ == "__main__":
    noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    n_train_list = [10, 30, 50, 70, 100]    # number of training samples for each label

    torch.manual_seed(123)
    np.random.seed(123)
    for n_train in n_train_list:
        n_val = int(n_train * 0.4)  # number of validation samples for each label
        generate_data(n_train, n_val, noise_prob_list)