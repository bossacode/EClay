import torch
from torch.distributions import Bernoulli
from sklearn.model_selection import train_test_split
import os
from utils.dtm import DTMLayer


def add_noise(x, p):
    """Replace points with random noise between 0 and 1 with probabilty p.

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


def gen_orbit(num_pts, rho):
    """Generate one orbit.

    Args:
        num_pts (int): Number of points in one orbit.
        rho (float): Parameter defining the dynamical system.

    Returns:
        torch.Tensor: Tensor of shape [num_pts, 2]
    """
    X = torch.zeros(num_pts, 2)
    X[0] = torch.rand(2)
    for i in range(1, num_pts):
        x, y = X[i-1]
        x_new = (x + rho*y*(1-y)) % 1
        y_new = (y + rho*x_new*(1-x_new)) % 1
        X[i] = torch.tensor([x_new, y_new])
    return X


def gen_orbit5K(rhos=[2.5, 3.5, 4.0, 4.1, 4.3], num_pts=1000, num_orbits_each=1000):
    """Generate entire ORBIT5K dataset.

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
    X_grid = torch.zeros(X.shape[0], int(1./by), int(1./by))
    for i in range(len(X)):
        orbit_int = torch.floor(X[i] / by).to(int) - (X[i] == (1./by)).to(int)
        for iPt in range(len(orbit_int)):
            X_grid[i][orbit_int[iPt][0], orbit_int[iPt][1]] += 1
    X_grid = 2 * torch.sigmoid(X_grid) - 1  
    return (X_grid)


def generate_data(val_size, test_size, noise_prob_list, rhos=[2.5, 3.5, 4.0, 4.1, 4.3], num_pts=1000):
    X, y = gen_orbit5K(rhos, num_pts, num_orbits_each=1000)
    # split test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123, shuffle=True, stratify=y)
    # split val data
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=123, shuffle=True, stratify=y_train)
    
    dtm = DTMLayer(m0=0.01)

    processed_dir = "./dataset/processed/"

    for p in noise_prob_list:
        dir_name = f"{processed_dir}noise_" + str(int(p * 100)).zfill(2) + "/"
        os.makedirs(dir_name, exist_ok=True)
        X_tr_noise = add_noise(X_tr, p)
        X_tr_grid = pc2grid(X_tr_noise).unsqueeze(1)
        X_tr_dtm = dtm(X_tr_noise).unsqueeze(1)
        torch.save((X_tr_noise, X_tr_grid, X_tr_dtm, y_tr), f=dir_name + f"train.pt")


        X_val_noise = add_noise(X_val, p)
        X_val_grid = pc2grid(X_val_noise).unsqueeze(1)
        X_val_dtm = dtm(X_val_noise).unsqueeze(1)
        torch.save((X_val_noise, X_val_grid, X_val_dtm, y_val), f=dir_name + f"val.pt")

        X_test_noise = add_noise(X_test, p)
        X_test_grid = pc2grid(X_test_noise).unsqueeze(1)
        X_test_dtm = dtm(X_test_noise).unsqueeze(1)
        torch.save((X_test_noise, X_test_grid, X_test_dtm, y_test), f=dir_name + f"test.pt")


if __name__ == "__main__":
    noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2]   # noise probabilities
    # n_train_list = [100, 300, 500, 700, 1000]     # number of training samples

    torch.manual_seed(123)
    generate_data(val_size=0.1, test_size=0.3, noise_prob_list=noise_prob_list)