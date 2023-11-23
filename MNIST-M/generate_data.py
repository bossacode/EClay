import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from PIL import Image
import random
import os


N = 30  # number of samples for each category


def corrupt_noise(X, corrupt_prob, noise_prob):
    X_corrupt = torch.where(torch.bernoulli(torch.zeros(X.shape), p=corrupt_prob).bool(),
                            0,
                            X)
    X_corrupt_noise = torch.where(((X < 0.01) * torch.bernoulli(torch.zeros(X.shape), p=noise_prob)).bool(),
                                  torch.rand(X.shape),
                                  X_corrupt)
    return X_corrupt_noise


def generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file, dir="./generated_data/"):
    transform = ToTensor()
    
    # train data
    train_data_list = []
    train_label_list = []
    for label_dir in os.listdir("training/"):
        label = int(label_dir)
        sampled_img = random.sample(os.listdir("training/" + label_dir), N)
        for img_name in sampled_img:
            img = Image.open("training/" + label_dir + "/" + img_name)
            train_data_list.append(transform(img))
            train_label_list.append(label)
    x_train = torch.stack(train_data_list)      # data shape: (10*N, 3, 32, 32)
    y_train = torch.tensor(train_label_list)

    # test data
    test_data_list = []
    test_label_list = []
    for label_dir in os.listdir("testing/"):
        label = int(label_dir)
        for img_name in os.listdir("testing/" + label_dir):
            img = Image.open("testing/" + label_dir + "/" + img_name)
            test_data_list.append(transform(img))
            test_label_list.append(label)
    x_test = torch.stack(test_data_list)        # data shape: (9000, 3, 32, 32)
    y_test = torch.tensor(test_label_list)

    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save((y_train, y_test), dir + y_file)

    len_cn = len(corrupt_prob_list)
    for i in range(len_cn):
        x_train_noise = corrupt_noise(x_train, corrupt_prob_list[i], noise_prob_list[i])
        x_test_noise = corrupt_noise(x_test, corrupt_prob_list[i], noise_prob_list[i])
        torch.save((x_train_noise, x_test_noise), dir + x_file_list[i])


if __name__ == "__main__":
    # corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    corrupt_prob_list = [0.0, 0.1, 0.2]
    noise_prob_list = [0.0, 0.1, 0.2]
    
    len_cn = len(corrupt_prob_list)
    file_cn_list = [None] * len_cn
    for i in range(len_cn):
        file_cn_list[i] = str(int(corrupt_prob_list[i] * 100)).zfill(2) + "_" + str(int(noise_prob_list[i] * 100)).zfill(2)
        
    x_file_list = ["x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
    y_file = "y.pt"

    torch.manual_seed(123)
    random.seed(123)
    generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file)