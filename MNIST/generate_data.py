import torch
from torch.distributions import Bernoulli
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import os


# def corrupt_noise(x, corrupt_p, noise_p):
#     # corruption process
#     corrupt_dist = Bernoulli(probs=corrupt_p)
#     x_corrupt = torch.where(corrupt_dist.sample(x.shape).bool(),
#                             0,
#                             x)
    
#     # noise process
#     noise_dist = Bernoulli(probs=noise_p)
#     x_corrupt_noise = torch.where(((x < 0.01) * noise_dist.sample(x.shape)).bool(),
#                                   torch.rand(x.shape),
#                                   x_corrupt)
#     return x_corrupt_noise


# def generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file, dir="./generated_data/"):
#     # train data shape: (60000,28,28)
#     train_data = MNIST(root="./raw_data",
#                        train=True,
#                        download=True,
#                        transform=ToTensor())
    
#     # test data shape: (10000,28,28) 
#     test_data = MNIST(root="./raw_data",
#                       train=False,
#                       download=True,
#                       transform=ToTensor())
    
#     # normalize
#     x_train = train_data.data / 255
#     x_test = test_data.data / 255

#     y_train = train_data.targets
#     y_test = test_data.targets

#     # sample N training data for each label
#     idx_list = []
#     for label in y_train.unique():
#         sampled_idx = np.random.choice(torch.where(y_train == label)[0], N)
#         idx_list.append(sampled_idx)
#     idx = np.concatenate(idx_list)
#     x_train = x_train[idx]
#     y_train = y_train[idx]

#     # ind = torch.randint(low=0, high=len(train_data.data), size=(N,))
#     # x_train = x_train[ind]
#     # y_train = y_train[ind]

#     # add channel dimension: (N,C,H,W)
#     x_train.unsqueeze_(1)
#     x_test.unsqueeze_(1)

#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     torch.save((y_train, y_test), dir + y_file)

#     len_cn = len(corrupt_prob_list)
#     for i in range(len_cn):
#         x_train_noise = corrupt_noise(x_train, corrupt_prob_list[i], noise_prob_list[i])
#         x_test_noise = corrupt_noise(x_test, corrupt_prob_list[i], noise_prob_list[i])
#         torch.save((x_train_noise, x_test_noise), dir + x_file_list[i])


n_train = 10                   # number of training samples for each label
n_val = int(n_train * 0.3)     # number of validation samples for each label


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
    train_data = MNIST(root="./raw_data", train=True, download=True, transform=ToTensor())  # shape: (60000, 28, 28)
    test_data = MNIST(root="./raw_data", train=False, download=True, transform=ToTensor())  # shape: (10000, 28, 28) 
    
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