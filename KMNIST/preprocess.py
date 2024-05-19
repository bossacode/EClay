import torch
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor, Compose
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import FaceToEdge
import vedo
from dtm import WeightedDTMLayer
from eclay import CubECC2d
from pllay import CubPL2d
import numpy as np
import os


def add_noise(x, p):
    """Replace pixels with random noise between 0 and 1 with probabilty p.

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
    """Tranform data using DTM.

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
    """Transform data to Euler Characteristic Curve.

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
    """Transform data to Persistent Landscape.

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


###############used for creating dataset for DECT############################################
class Kmnist2Complex:
    def __init__(self):
        x_seq = torch.linspace(start=-0.5, end=0.5, steps=28)
        y_seq = torch.linspace(start=0.5, end=-0.5, steps=28)
        self.X, self.Y = torch.meshgrid(x_seq, y_seq, indexing="xy")

    def __call__(self, data):
        """_summary_

        Args:
            data (tuple): Tuple of tensors: (img, label)

        Returns:
            torch_geometric.data.data.Data: _description_
        """
        X, y = data
        idx = torch.nonzero(X, as_tuple=True)
        pc = torch.vstack([self.X[idx], self.Y[idx]]).T
        dly = vedo.delaunay2d(pc, mode="xy", alpha=0.03).c("w").lc("o").lw(1)
        return Data(x=torch.tensor(dly.vertices),
                    face=torch.tensor(dly.cells, dtype=torch.long).T,
                    y=y)


class CenterTransform:
    def __call__(self, data):
        """
        Args:
            data (torch_geometric.data.data.Data): _description_

        Returns:
            _type_: _description_
        """
        data.x -= data.x.mean()
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


class KmnistDataset(InMemoryDataset):
    def __init__(self, img, label, noise_prob, data_size=None, mode="train", root="./dataset/", transform=None,
                 pre_transform=Compose([Kmnist2Complex(), FaceToEdge(remove_faces=False), CenterTransform()]), pre_filter=None):
        """_summary_

        Args:
            img (torch.Tensor): Image data of shape [N, 1, 28, 28]
            label (torch.Tensor): Label data of shape [N,]
            mode (str, optional): _description_. Defaults to "train".
            root (str, optional): _description_. Defaults to "./dataset/".
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to Compose([Mnist2PointCloud(), FaceToEdge(remove_faces=False), CenterTransform()]).
            pre_filter (_type_, optional): _description_. Defaults to None.
        """
        self.X, self.y = img, label
        self.noise_prob = noise_prob
        self.data_size = data_size
        self.mode = mode
        self.noise_prob = noise_prob
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["MNIST"]

    @property
    def processed_file_names(self):
        if self.mode == "train":
            return [f"dect/noise_{str(int(self.noise_prob * 100)).zfill(2)}/data_{self.data_size}/train.pt"]
        elif self.mode == "val":
            return [f"dect/noise_{str(int(self.noise_prob * 100)).zfill(2)}/data_{self.data_size}/val.pt"]
        else:
            return [f"dect/noise_{str(int(self.noise_prob * 100)).zfill(2)}/test.pt"]

    def process(self):
        data_list = [self.pre_transform((img[0], lab)) for  img, lab in zip(self.X, self.y)]
        self.save(data_list, self.processed_paths[0])
###########################################################################################################################


def generate_data(n_train_each_list, noise_prob_list, val_size):
    """_summary_

    Args:
        n_train_each_list (list): Number of training data to sample for each label.
        val_size (float): Size of validation set in proportion to train set.
        noise_prob(list): Noise probabilities.
    """
    train_data = KMNIST(root="./dataset/raw/", train=True, download=True, transform=ToTensor())  # shape: (60000, 28, 28)
    test_data = KMNIST(root="./dataset/raw/", train=False, download=True, transform=ToTensor())  # shape: (10000, 28, 28)
    
    # normalize and add channel dimension: (N, C, H, W)
    x_train = (train_data.data / 255).unsqueeze(1)
    y_train = train_data.targets
    x_test = (test_data.data / 255).unsqueeze(1)
    y_test = test_data.targets

    dtm005 = WeightedDTMLayer(m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
    # use V-construction
    # ecc_dtm005 = CubECC2d(as_vertices=True, sublevel=True, size=[28, 28], interval=[0.02, 0.28], steps=32)
    # pl_dtm005 = CubPL2d(as_vertices=True, sublevel=True, interval=[0.02, 0.28], steps=32, K_max=2, dimensions=[0, 1])
    # use T-construction
    ecc_dtm005 = CubECC2d(as_vertices=False, sublevel=True, size=[28, 28], interval=[0.03, 0.34], steps=32)
    pl_dtm005 = CubPL2d(as_vertices=False, sublevel=True, interval=[0.03, 0.34], steps=32, K_max=2, dimensions=[0, 1])

    dtm02 = WeightedDTMLayer(m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
    # use V-construction
    # ecc_dtm02 = CubECC2d(as_vertices=True, sublevel=True, size=[28, 28], interval=[0.06, 0.29], steps=32)
    # pl_dtm02 = CubPL2d(as_vertices=True, sublevel=True, interval=[0.06, 0.29], steps=32, K_max=3, dimensions=[0, 1])
    # use T-construction
    ecc_dtm02 = CubECC2d(as_vertices=False, sublevel=True, size=[28, 28], interval=[0.06, 0.35], steps=32)
    pl_dtm02 = CubPL2d(as_vertices=False, sublevel=True, interval=[0.06, 0.35], steps=32, K_max=2, dimensions=[0, 1])

    for noise_prob in noise_prob_list:
        eclayr_dir = f"./dataset/processed/eclayr/noise_" + str(int(noise_prob * 100)).zfill(2) + "/"
        dect_dir = f"./dataset/processed/dect/noise_" + str(int(noise_prob * 100)).zfill(2) + "/"
        for n_train_each in n_train_each_list:
            # sample "n_train_each" training data and "n_train_each * val_size" validation data for each label
            n_val_each = int(n_train_each * val_size)
            train_idx_list = []
            val_idx_list = []
            labels = y_train.unique()
            for i in labels:
                idx = torch.where(y_train == i)[0]
                sampled_idx = np.random.choice(idx, size=n_train_each + n_val_each, replace=False)
                train_idx_list.append(sampled_idx[:n_train_each])
                val_idx_list.append(sampled_idx[n_train_each:])
            train_idx = np.concatenate(train_idx_list)
            val_idx = np.concatenate(val_idx_list)
            x_tr_sampled, y_tr_sampled = x_train[train_idx], y_train[train_idx]
            x_val_sampled, y_val_sampled = x_train[val_idx], y_train[val_idx]
            
            # train data
            x_tr_noise = add_noise(x_tr_sampled, noise_prob)
            x_tr_dtm005 = dtm_transform(x_tr_noise, dtm005)
            ecc_tr_dtm005 = ecc_transform(x_tr_dtm005, ecc_dtm005)
            pl_tr_dtm005 = pl_transform(x_tr_dtm005, pl_dtm005)
            x_tr_dtm02 = dtm_transform(x_tr_noise, dtm02)
            ecc_tr_dtm02 = ecc_transform(x_tr_dtm02, ecc_dtm02)
            pl_tr_dtm02 = pl_transform(x_tr_dtm02, pl_dtm02)
            
            # validation data
            x_val_noise = add_noise(x_val_sampled, noise_prob)
            x_val_dtm005 = dtm_transform(x_val_noise, dtm005)
            ecc_val_dtm005 = ecc_transform(x_val_dtm005, ecc_dtm005)
            pl_val_dtm005 = pl_transform(x_val_dtm005, pl_dtm005)
            x_val_dtm02 = dtm_transform(x_val_noise, dtm02)
            ecc_val_dtm02 = ecc_transform(x_val_dtm02, ecc_dtm02)
            pl_val_dtm02 = pl_transform(x_val_dtm02, pl_dtm02)

            os.makedirs(eclayr_dir + f"data_{len(y_tr_sampled)}/", exist_ok=True)
            torch.save((x_tr_noise, ecc_tr_dtm005, ecc_tr_dtm02, pl_tr_dtm005, pl_tr_dtm02, y_tr_sampled), eclayr_dir + f"data_{len(y_tr_sampled)}/train.pt")
            torch.save((x_val_noise, ecc_val_dtm005, ecc_val_dtm02, pl_val_dtm005, pl_val_dtm02, y_val_sampled), eclayr_dir + f"data_{len(y_tr_sampled)}/val.pt")

            # dect train and val data
            os.makedirs(dect_dir + f"data_{len(y_tr_sampled)}/", exist_ok=True)
            KmnistDataset(x_tr_noise, y_tr_sampled, noise_prob, len(y_tr_sampled), mode="train")
            KmnistDataset(x_val_noise, y_val_sampled, noise_prob, len(y_tr_sampled), mode="val")

        # test data
        x_test_noise = add_noise(x_test, noise_prob)
        x_test_dtm005 = dtm_transform(x_test_noise, dtm005)
        ecc_test_dtm005 = ecc_transform(x_test_dtm005, ecc_dtm005)
        pl_test_dtm005 = pl_transform(x_test_dtm005, pl_dtm005)
        x_test_dtm02 = dtm_transform(x_test_noise, dtm02)
        ecc_test_dtm02 = ecc_transform(x_test_dtm02, ecc_dtm02)
        pl_test_dtm02 = pl_transform(x_test_dtm02, pl_dtm02)
        torch.save((x_test_noise, ecc_test_dtm005, ecc_test_dtm02, pl_test_dtm005, pl_test_dtm02, y_test), eclayr_dir + "test.pt")

        # dect test data
        KmnistDataset(x_test_noise, y_test, noise_prob, mode="test")


if __name__ == "__main__":
    n_train_each_list = [10, 30, 50, 70, 100]           # number of training samples for each label
    # n_train_each_list = [10]
    noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25] # noise probabilities
    # noise_prob_list = [0.0]

    torch.manual_seed(123)
    np.random.seed(123)
    generate_data(n_train_each_list, noise_prob_list, val_size=0.4)