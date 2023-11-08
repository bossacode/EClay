import torch
from torch.distributions import Binomial
from torchvision.transforms import Resize, ToTensor, Compose
import pandas as pd
from PIL import Image
from random import sample, seed
import os


N = 200
label_map = {'nv':0, 'mel':1, 'bkl':2, 'df':3, 'akiec':4, 'bcc':5, 'vasc':6}

def corrupt_noise(X, corrupt_prob, noise_prob):
    X_corrupt = torch.where(Binomial(1, probs=corrupt_prob).sample(X.shape).bool(),
                            0,
                            X)
    X_corrupt_noise = torch.where(((X < 0.01) * Binomial(1, probs=corrupt_prob).sample(X.shape)).bool(),
                                  torch.rand(X.shape),
                                  X_corrupt)
    return X_corrupt_noise


def generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file, dir="./generated_data/"):
    transform= Compose([Resize((224, 224)),
                        ToTensor()])
    
    # only use part1 (5000 data) for now
    meta_data = pd.read_csv("HAM10000_metadata").sort_values(by='image_id').iloc[:5000]
    data_size = len(meta_data)
    train_ind = sample(range(data_size), N)
    test_ind = [i for i in range(data_size) if i not in train_ind]
    
    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    for i in range(data_size):
        img_id = meta_data['image_id'].iloc[i]
        img = Image.open("HAM10000_images_part_1/" + img_id + ".jpg")
        y_label = meta_data['dx'].iloc[i]
        if i in train_ind:
            x_train_list.append(transform(img))
            y_train_list.append(label_map[y_label])
        else:
            x_test_list.append(transform(img))
            y_test_list.append(label_map[y_label])
    
    # train data shape: (60000,28,28)
    x_train = torch.stack(x_train_list)
    y_train = torch.tensor(y_train_list)
    x_test = torch.stack(x_test_list)
    y_test = torch.tensor(y_test_list)

    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save((y_train, y_test), dir + y_file)

    len_cn = len(corrupt_prob_list)
    for image_id in range(len_cn):
        x_train_noise = corrupt_noise(x_train, corrupt_prob_list[image_id], noise_prob_list[image_id])
        x_test_noise = corrupt_noise(x_test, corrupt_prob_list[image_id], noise_prob_list[image_id])
        torch.save((x_train_noise, x_test_noise), dir + x_file_list[image_id])


if __name__ == "__main__":
    corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    len_cn = len(corrupt_prob_list)
    file_cn_list = [None] * len_cn
    for i in range(len_cn):
        file_cn_list[i] = str(int(corrupt_prob_list[i] * 100)).zfill(2) + "_" + str(int(noise_prob_list[i] * 100)).zfill(2)
        
    x_file_list = ["x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
    y_file = "y.pt"

    torch.manual_seed(123)
    seed(123)
    generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file)