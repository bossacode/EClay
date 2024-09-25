import torch
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from PIL import Image
from random import sample, seed
import os
import sys
sys.path.append("../")
from utils.preprocess import cn_transform, dtm_transform

train_root = "./dataset/MNIST-M/training/"
test_root = "./dataset/MNIST-M/testing/"
lims = [[-0.5, 0.5], [-0.5, 0.5]]
size = [32, 32]
transform = ToTensor()


def gen_sampled_data(train_size_list, val_size=0.3, num_labels=10):
    """_summary_

    Args:
        train_size_list (list): List containing number of train data to sample.
        val_size (float, optional): Proportion of validation split. Defaults to 0.3.
        num_labels (int, optional): Number of unique labels. Defaults to 10.
    """
    processed_data_dir = "./dataset/processed/train_size/"
    # train data shape: (59000, 3, 32, 32)
    for train_size in train_size_list:
        # sample "train_size" number of train data with equal proportion per label
        x_tr_list = []
        y_tr_list = []
        for label in os.listdir(train_root):
            img_list = os.listdir(train_root + label)
            sampled_img_list = sample(img_list, int(train_size / num_labels))
            for img in sampled_img_list:
                data = Image.open(train_root + label + "/" + img)
                x_tr_list.append(transform(data))
                y_tr_list.append(int(label))

        x_tr_sampled = torch.stack(x_tr_list)
        y_tr_sampled = torch.tensor(y_tr_list)

        # split train and validation data
        x_tr, x_val, y_tr, y_val = train_test_split(x_tr_sampled, y_tr_sampled, test_size=val_size, random_state=123, shuffle=True, stratify=y_tr_sampled)

        # apply DTM on train data
        x_tr_dtm005 = dtm_transform(x_tr, m0=0.05, lims=lims, size=size)
        x_tr_dtm02 = dtm_transform(x_tr, m0=0.2, lims=lims, size=size)

        # apply DTM on validation data
        x_val_dtm005 = dtm_transform(x_val, m0=0.05, lims=lims, size=size)
        x_val_dtm02 = dtm_transform(x_val, m0=0.2, lims=lims, size=size)
        
        # save train and validation data
        os.makedirs(processed_data_dir + f"{train_size}/", exist_ok=True)
        torch.save((x_tr, x_tr_dtm005, x_tr_dtm02, y_tr), f=processed_data_dir + f"{train_size}/train.pt")
        torch.save((x_val, x_val_dtm005, x_val_dtm02, y_val), f=processed_data_dir + f"{train_size}/val.pt")

    # test data shape: (9000, 3, 32, 32)
    x_test_list = []
    y_test_list = []
    for label in os.listdir(test_root):
        img_list = os.listdir(test_root + label)
        for img in img_list:
            data = Image.open(test_root + label + "/" + img)
            x_test_list.append(transform(data))
            y_test_list.append(int(label))

    x_test = torch.stack(x_test_list)
    y_test = torch.tensor(y_test_list)
    
    # apply DTM on test data
    x_test_dtm005 = dtm_transform(x_test, m0=0.05, lims=lims, size=size)
    x_test_dtm02 = dtm_transform(x_test, m0=0.2, lims=lims, size=size)

    # save test data
    torch.save((x_test, x_test_dtm005, x_test_dtm02, y_test), f=processed_data_dir + "test.pt")


# def gen_noise_data(cn_prob_list, dir_path):
#     """_summary_

#     Args:
#         cn_prob_list (list): List containing corruption and noise probabilities.
#         dir_path (int): Path to the directory that contains the sampled train and validation data to which we want to add noise.
#     """
#     # load sampled train and validation data
#     x_tr, _, _, y_tr = torch.load(dir_path + "/train.pt", weights_only=True)    # shape: (N_train, 1, 28, 28)
#     x_val, _, _, y_val = torch.load(dir_path + "/val.pt", weights_only=True)    # shape: (N_val, 1, 28, 28)
    
#     # load test data
#     test_data = KMNIST(root="./dataset/raw/", train=False, download=True, transform=ToTensor()) # shape: (10000, 28, 28)
#     x_test = (test_data.data / 255).unsqueeze(1)
#     y_test = test_data.targets

#     for p in cn_prob_list:
#         # apply DTM on corrupted and noised train data
#         x_tr_cn = cn_transform(x_tr, p)
#         x_tr_cn_dtm005 = dtm_transform(x_tr_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
#         x_tr_cn_dtm02 = dtm_transform(x_tr_cn, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

#         # apply DTM on corrupted and noised validation data
#         x_val_cn = cn_transform(x_val, p)
#         x_val_cn_dtm005 = dtm_transform(x_val_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
#         x_val_cn_dtm02 = dtm_transform(x_val_cn, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

#         # apply DTM on corrupted and noised test data
#         x_test_cn = cn_transform(x_test, p)
#         x_test_cn_dtm005 = dtm_transform(x_test_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
#         x_test_cn_dtm02 = dtm_transform(x_test_cn, m0=0.2, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

#         # save train, validation and test data
#         dir = "./dataset/processed/cn_prob/" + str(int(p * 100)).zfill(2) + "/"
#         os.makedirs(dir, exist_ok=True)
#         torch.save((x_tr_cn, x_tr_cn_dtm005, x_tr_cn_dtm02, y_tr), f=dir + "train.pt")
#         torch.save((x_val_cn, x_val_cn_dtm005, x_val_cn_dtm02, y_val), f=dir + "val.pt")
#         torch.save((x_test_cn, x_test_cn_dtm005, x_test_cn_dtm02, y_test), f=dir + "test.pt")


if __name__ == "__main__":
    train_size_list = [300, 500, 700, 1000, 10000]    # training sample sizes
    cn_prob_list = [0.05, 0.1, 0.15, 0.2, 0.25]     # corruption and noise probabilities
    val_size=0.3                                    # proportion of validation split

    seed(123)
    torch.manual_seed(123)
    gen_sampled_data(train_size_list, val_size, num_labels=10)
    # gen_noise_data(cn_prob_list, dir_path="./dataset/processed/train_size/500/")  # "gen_sampled_data" must be preceeded to create directory containing sampled data before running "gen_noise_data"