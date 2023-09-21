import os
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def corrupt_noise(X, corrupt_prob, noise_prob):
    X_crpt = np.where(np.random.binomial(n=1, p=corrupt_prob, size=X.shape),
                      0,
                      X)
    X_crpt_noise = np.where((X < 0.01) * np.random.binomial(n=1, p=noise_prob, size=X.shape),
                            np.random.uniform(0.0, 1.0, size=X.shape),
                            X_crpt)
    return torch.from_numpy(X_crpt_noise).to(torch.float32)


def mnist_generate_data(N, corrupt_prob_list, noise_prob_list, x_original_file_list, y_file, dir='./generated_data/'):
    # train data shape: (60000,28,28)
    train_data = MNIST(root='./raw_data',
                       train=True,
                       download=True,
                       transform=ToTensor())
    
    # test data shape: (10000,28,28) 
    test_data = MNIST(root='./raw_data',
                      train=False,
                      download=True,
                      transform=ToTensor())
    
    # normalize
    x_train = train_data.data / 255.
    x_test = test_data.data / 255.

    y_train = train_data.targets
    y_test = test_data.targets

    # sample N training data
    ind = np.random.randint(low=0, high=len(train_data.data), size=N)
    x_train = x_train[ind]
    y_train = y_train[ind]

    # add channel dimension: (N,C,H,W)
    x_train.unsqueeze_(1)
    x_test.unsqueeze_(1)

    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save((y_train, y_test), dir + y_file)

    nCn = len(corrupt_prob_list)
    for iCn in range(nCn):
        x_train_noise = corrupt_noise(x_train, corrupt_prob_list[iCn], noise_prob_list[iCn])
        x_test_noise = corrupt_noise(x_test, corrupt_prob_list[iCn], noise_prob_list[iCn])
        torch.save((x_train_noise, x_test_noise), dir + x_original_file_list[iCn])


# main code
if __name__ == '__main__':
    corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    nCn = len(corrupt_prob_list)
    file_cn_list = [None] * nCn
    for iCn in range(nCn):
        file_cn_list[iCn] = str(int(corrupt_prob_list[iCn] * 100)).zfill(2) + \
            '_' + str(int(noise_prob_list[iCn] * 100)).zfill(2)
        
    x_original_file_list = ['mnist_x_original_' + file_cn_list[iCn] + '.pt' for iCn in range(nCn)]
    y_file = 'mnist_y.pt'

    N = 1000

    np.random.seed(0)
    mnist_generate_data(N, corrupt_prob_list, noise_prob_list, x_original_file_list, y_file)