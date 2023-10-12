import torch
from torch.distributions import Binomial
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
import os

def corrupt_noise(X, corrupt_prob, noise_prob):
    X_corrupt = torch.where(Binomial(1, probs=corrupt_prob).sample(X.shape).bool(),
                            0,
                            X)
    X_corrupt_noise = torch.where(((X < 0.01) * Binomial(1, probs=corrupt_prob).sample(X.shape)).bool(),
                                  torch.rand(X.shape),
                                  X_corrupt)
    return X_corrupt_noise


def generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file, dir="./generated_data/"):
    # train data shape: (60000,28,28)
    train_data = KMNIST(root="./raw_data",
                        train=True,
                        download=True,
                        transform=ToTensor())
    
    # test data shape: (10000,28,28) 
    test_data = KMNIST(root="./raw_data",
                        train=False,
                        download=True,
                        transform=ToTensor())
    
    # normalize
    x_train = train_data.data / 255.
    x_test = test_data.data / 255.

    y_train = train_data.targets
    y_test = test_data.targets

    # sample N training data
    ind = torch.randint(low=0, high=len(y_train), size=(N,))
    x_train = x_train[ind]
    y_train = y_train[ind]

    # add channel dimension: (N,C,H,W)
    x_train.unsqueeze_(1)
    x_test.unsqueeze_(1)

    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save((y_train, y_test), dir + y_file)

    len_cn = len(corrupt_prob_list)
    for i in range(len_cn):
        x_train_cn = corrupt_noise(x_train, corrupt_prob_list[i], noise_prob_list[i])
        x_test_cn = corrupt_noise(x_test, corrupt_prob_list[i], noise_prob_list[i])
        torch.save((x_train_cn, x_test_cn), dir + x_file_list[i])


if __name__ == "__main__":
    corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    len_cn = len(corrupt_prob_list)
    file_cn_list = [None] * len_cn
    for i in range(len_cn):
        file_cn_list[i] = str(int(corrupt_prob_list[i] * 100)).zfill(2) + "_" + str(int(noise_prob_list[i] * 100)).zfill(2)
        
    x_file_list = ["x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
    y_file = "y.pt"

    N = 1000

    torch.manual_seed(123)
    generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file)