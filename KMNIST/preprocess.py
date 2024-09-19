import torch
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
sys.path.append("../")
from utils.preprocess import cn_transform, dtm_transform


def gen_sampled_data(train_size_list, val_size=0.3, num_labels=10):
    """_summary_

    Args:
        train_size_list (list): List containing number of train data to sample.
        val_size (float, optional): Proportion of validation split. Defaults to 0.3.
        num_labels (int, optional): Number of unique labels. Defaults to 10.
    """
    train_data = KMNIST(root="./dataset/raw/", train=True, download=True, transform=ToTensor()) # shape: (60000, 28, 28)
    test_data = KMNIST(root="./dataset/raw/", train=False, download=True, transform=ToTensor()) # shape: (10000, 28, 28)
    
    # normalize and add channel dimension: (N, 1, H, W)
    x_train = (train_data.data / 255).unsqueeze(1)
    y_train = train_data.targets
    x_test = (test_data.data / 255).unsqueeze(1)
    y_test = test_data.targets

    dir = "./dataset/processed/train_size/"
    for train_size in train_size_list:
        # sample "train_size" number of train data with equal proportion per label
        train_idx = []
        labels = y_train.unique()
        for i in labels:
            idx = torch.where(y_train == i)[0]
            sampled_idx = np.random.choice(idx, size=int(train_size / num_labels), replace=False)
            train_idx.extend(sampled_idx)
        x_tr_sampled, y_tr_sampled = x_train[train_idx], y_train[train_idx]

        # split train and validation data
        x_tr, x_val, y_tr, y_val = train_test_split(x_tr_sampled, y_tr_sampled, test_size=val_size, random_state=123, shuffle=True, stratify=y_tr_sampled)
        
        # apply DTM on train data
        x_tr_dtm005 = dtm_transform(x_tr, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
        x_tr_dtm02 = dtm_transform(x_tr, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

        # apply DTM on validation data
        x_val_dtm005 = dtm_transform(x_val, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
        x_val_dtm02 = dtm_transform(x_val, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
        
        # save train and validation data
        os.makedirs(dir + f"{train_size}/", exist_ok=True)
        torch.save((x_tr, x_tr_dtm005, x_tr_dtm02, y_tr), f=dir + f"{train_size}/train.pt")
        torch.save((x_val, x_val_dtm005, x_val_dtm02, y_val), f=dir + f"{train_size}/val.pt")
    
    # apply DTM on test data
    x_test_dtm005 = dtm_transform(x_test, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
    x_test_dtm02 = dtm_transform(x_test, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

    # save test data
    torch.save((x_test, x_test_dtm005, x_test_dtm02, y_test), f=dir + "test.pt")


def gen_noise_data(cn_prob_list, dir_path):
    """_summary_

    Args:
        cn_prob_list (list): List containing corruption and noise probabilities.
        dir_path (int): Path to the directory that contains the sampled train and validation data to which we want to add noise.
    """
    # load sampled train and validation data
    x_tr, _, _, y_tr = torch.load(dir_path + "/train.pt", weights_only=True)    # shape: (N_train, 1, 28, 28)
    x_val, _, _, y_val = torch.load(dir_path + "/val.pt", weights_only=True)    # shape: (N_val, 1, 28, 28)
    
    # load test data
    test_data = KMNIST(root="./dataset/raw/", train=False, download=True, transform=ToTensor()) # shape: (10000, 28, 28)
    x_test = (test_data.data / 255).unsqueeze(1)
    y_test = test_data.targets

    for p in cn_prob_list:
        # apply DTM on corrupted and noised train data
        x_tr_cn = cn_transform(x_tr, p)
        x_tr_cn_dtm005 = dtm_transform(x_tr_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
        x_tr_cn_dtm02 = dtm_transform(x_tr_cn, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

        # apply DTM on corrupted and noised validation data
        x_val_cn = cn_transform(x_val, p)
        x_val_cn_dtm005 = dtm_transform(x_val_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
        x_val_cn_dtm02 = dtm_transform(x_val_cn, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

        # apply DTM on corrupted and noised test data
        x_test_cn = cn_transform(x_test, p)
        x_test_cn_dtm005 = dtm_transform(x_test_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
        x_test_cn_dtm02 = dtm_transform(x_test_cn, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

        # save train, validation and test data
        dir = "./dataset/processed/cn_prob/" + str(int(p * 100)).zfill(2) + "/"
        os.makedirs(dir, exist_ok=True)
        torch.save((x_tr_cn, x_tr_cn_dtm005, x_tr_cn_dtm02, y_tr), f=dir + "train.pt")
        torch.save((x_val_cn, x_val_cn_dtm005, x_val_cn_dtm02, y_val), f=dir + "val.pt")
        torch.save((x_test_cn, x_test_cn_dtm005, x_test_cn_dtm02, y_test), f=dir + "test.pt")


if __name__ == "__main__":
    train_size_list = [300, 500, 700, 1000, 10000]    # training sample sizes
    cn_prob_list = [0.05, 0.1, 0.15, 0.2, 0.25]     # corruption and noise probabilities
    val_size=0.3                                    # proportion of validation split

    np.random.seed(123)
    torch.manual_seed(123)
    gen_sampled_data(train_size_list, val_size, num_labels=10)
    gen_noise_data(cn_prob_list, dir_path="./dataset/processed/train_size/500/")  # "gen_sampled_data" must be preceeded to create directory containing sampled data before running "gen_noise_data"