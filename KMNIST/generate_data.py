import torch
from torch.distributions import Bernoulli
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
import numpy as np
import os

n_train = 10                   # number of training samples for each label
n_val = int(n_train * 0.4)     # number of validation samples for each label


def add_noise(x, p):
    dist = Bernoulli(probs=p)
    x_noise = torch.where(dist.sample(x.shape).bool(),
                          torch.rand(x.shape),
                          x)
    return x_noise


def generate_data(n_train, n_val, noise_prob_list):
    """_summary_

    Args:
        n_train (int): _description_
        n_val (int): _description_
        noise_prob_list (list): _description_
    """
    train_data = KMNIST(root="./raw_data", train=True, download=True, transform=ToTensor())  # shape: (60000, 28, 28)
    test_data = KMNIST(root="./raw_data", train=False, download=True, transform=ToTensor())  # shape: (10000, 28, 28) 
    
    # normalize and add channel dimension: (N, C, H, W)
    x_train = (train_data.data / 255).unsqueeze(1)
    y_train = train_data.targets
    x_test = (test_data.data / 255).unsqueeze(1)
    y_test = test_data.targets

    # sample "n_train" training data and "n_val" validation data for each label
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

    for p in noise_prob_list:
        dir_name = "generated_data/noise_" + str(int(p * 100)).zfill(2) + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        x_train_noise = add_noise(x_train_sampled, p)
        x_val_noise = add_noise(x_val_sampled, p)
        x_test_noise = add_noise(x_test, p)
        torch.save((x_train_noise, y_train_sampled), dir_name + "train.pt")
        torch.save((x_val_noise, y_val_sampled), dir_name + "val.pt")
        torch.save((x_test_noise, y_test), dir_name + "test.pt")


if __name__ == "__main__":
    # noise_prob_list = [0.0]
    noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    torch.manual_seed(123)
    np.random.seed(123)
    generate_data(n_train, n_val, noise_prob_list)