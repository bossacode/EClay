import sys
sys.path.append("../")
import torch
from torch.distributions import Bernoulli
from sklearn.model_selection import train_test_split
import os
from utils.preprocess import noise, dtm_transform


def gen_orbit(num_pts, rho):
    """Generate one orbit.

    Args:
        num_pts (int): Number of points in one orbit.
        rho (float): Parameter defining the dynamical system.

    Returns:
        torch.Tensor: Tensor of shape [num_pts, 2]
    """
    X = torch.zeros(num_pts, 2)
    x, y = torch.rand(1).item(), torch.rand(1).item()
    for i in range(num_pts):
        x = (x + rho * y * (1-y)) % 1
        y = (y + rho * x * (1-x)) % 1
        X[i] = torch.tensor([x, y])
    return X


def gen_orbits(rhos=[2.5, 3.5, 4.0, 4.1, 4.3], num_pts=1000, num_orbits_each=1000):
    """Generate entire ORBIT dataset.

    Args:
        rhos (list, optional): List of parameters defining the dynamical system. Defaults to [2.5, 3.5, 4.0, 4.1, 4.3].
        num_pts (int, optional): Number of points in one orbit. Defaults to 1000.
        num_orbits_each (int, optional): Number of data for each label. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    X = torch.zeros(len(rhos)*num_orbits_each, num_pts, 2)
    y = []
    for label, rho in enumerate(rhos):
        for i in range(num_orbits_each):
            X[label*num_orbits_each + i] = gen_orbit(num_pts, rho)
            y.append(label)
    return X, torch.tensor(y)


def pc2grid(X, by = 0.025):
    """Map point clouds onto a grid.

    Args:
        X (torch.Tensor): Batch of point clouds. Tensor of shape [B, num_pts, D].
        by (float, optional): Interval between each grid point. Defaults to 0.025.

    Returns:
        X_grid (torch.Tensor): Batch of images made by mapping points clouds onto a grid. Tensor of shape [B, 1, grid_size, grid_size].
    """
    grid_size = int(1./by)  # size of one side of square grid
    X_grid = torch.zeros(X.shape[0], grid_size, grid_size)
    for i in range(len(X)):
        orbit_int = torch.floor(X[i] / by).to(int) - (X[i] == 1.).to(int)
        for iPt in range(len(orbit_int)):
            X_grid[i][orbit_int[iPt][0], grid_size-1-orbit_int[iPt][1]] += 1
    X_grid = 2 * torch.sigmoid(X_grid) - 1
    return X_grid.unsqueeze(1)


def gen_data(val_size, test_size, num_orbits_each_list, rhos=[2.5, 3.5, 4.0, 4.1, 4.3], num_pts=1000):
    dir = "./dataset/data_size/"
    for num_orbits_each in num_orbits_each_list:
        X, y = gen_orbits(rhos, num_pts, num_orbits_each=num_orbits_each)
        # split test data
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123, shuffle=True, stratify=y)
        # split val data
        x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=123, shuffle=True, stratify=y_train)
        
        # transform train data point cloud to image
        X_tr_grid = pc2grid(x_tr)
        # apply DTM on train data
        X_tr_dtm = dtm_transform(x_tr, m0=0.02, lims=[[0.0125, 0.9875], [0.0125, 0.9875]], size=[40, 40], weighted=False)
        
        # transform validation data point cloud to image
        X_val_grid = pc2grid(x_val)
        # apply DTM on validation data
        X_val_dtm = dtm_transform(x_val, m0=0.02, lims=[[0.0125, 0.9875], [0.0125, 0.9875]], size=[40, 40], weighted=False)
        
        # transform test data point cloud to image
        X_test_grid = pc2grid(x_test)
        # apply DTM on test data
        X_test_dtm = dtm_transform(x_test, m0=0.02, lims=[[0.0125, 0.9875], [0.0125, 0.9875]], size=[40, 40], weighted=False)
        
        # save train, validation, and test data
        os.makedirs(dir + f"{len(rhos)*num_orbits_each}/", exist_ok=True)
        torch.save((x_tr, X_tr_grid, X_tr_dtm, y_tr), f=dir + f"{len(rhos)*num_orbits_each}/train.pt")
        torch.save((x_val, X_val_grid, X_val_dtm, y_val), f=dir + f"{len(rhos)*num_orbits_each}/val.pt")
        torch.save((x_test, X_test_grid, X_test_dtm, y_test), f=dir + f"{len(rhos)*num_orbits_each}/test.pt")


def gen_noise_data(noise_prob_list, dir_path):
    # load sampled train and validation data
    x_tr, _, _, y_tr = torch.load(dir_path + "/train.pt", weights_only=True)
    x_val, _, _, y_val = torch.load(dir_path + "/val.pt", weights_only=True)
    x_test, _, _, y_test = torch.load(dir_path + "/test.pt", weights_only=True)

    dir = "./dataset/noise_prob/"
    for p in noise_prob_list:
        # add noise to train data
        X_tr_noise = noise(x_tr, p)
        # transform noised train data point cloud to image
        X_tr_grid = pc2grid(X_tr_noise)
        # apply DTM on noised train data
        X_tr_dtm = dtm_transform(X_tr_noise, m0=0.02, lims=[[0.0125, 0.9875], [0.0125, 0.9875]], size=[40, 40], weighted=False)
        
        # add noise to validation data
        X_val_noise = noise(x_val, p)
        # transform noised train validation point cloud to image
        X_val_grid = pc2grid(X_val_noise)
        # apply DTM on noised validation data
        X_val_dtm = dtm_transform(X_val_noise, m0=0.02, lims=[[0.0125, 0.9875], [0.0125, 0.9875]], size=[40, 40], weighted=False)

        # add noise to test data
        X_test_noise = noise(x_test, p)
        # transform noised test data point cloud to image
        X_test_grid = pc2grid(X_test_noise)
        # apply DTM on noised test data
        X_test_dtm = dtm_transform(X_test_noise, m0=0.02, lims=[[0.0125, 0.9875], [0.0125, 0.9875]], size=[40, 40], weighted=False)

        dir_name = dir + str(int(p * 100)).zfill(2) + "/"
        os.makedirs(dir_name, exist_ok=True)
        torch.save((X_tr_noise, X_tr_grid, X_tr_dtm, y_tr), f=dir_name + f"train.pt")
        torch.save((X_val_noise, X_val_grid, X_val_dtm, y_val), f=dir_name + f"val.pt")
        torch.save((X_test_noise, X_test_grid, X_test_dtm, y_test), f=dir_name + f"test.pt")


if __name__ == "__main__":
    num_orbits_each_list = [400, 500, 600, 800, 1000]   # sample (per label) sizes
    noise_prob_list = [0.05, 0.1, 0.15, 0.2]            # noise probabilities
    val_size=0.1                                        # proportion of validation split
    test_size=0.3                                       # proportion of test split

    torch.manual_seed(123)
    gen_data(val_size, test_size, num_orbits_each_list)
    gen_noise_data(noise_prob_list, dir_path="./dataset/data_size/5000/")