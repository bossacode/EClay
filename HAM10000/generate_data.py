import torch
from torch.distributions import Binomial
from torchvision.transforms import Resize, ToTensor, Compose
import pandas as pd
from PIL import Image
import os


N = 100  # number of samples for each category
label_map = {'nv':0, 'mel':1, 'bkl':2, 'df':3, 'akiec':4, 'bcc':5, 'vasc':6}

def corrupt_noise(X, corrupt_prob, noise_prob):
    X_corrupt = torch.where(torch.bernoulli(torch.zeros(X.shape), p=corrupt_prob).bool(),
                            0,
                            X)
    X_corrupt_noise = torch.where(((X < 0.01) * torch.bernoulli(torch.zeros(X.shape), p=noise_prob)).bool(),
                                  torch.rand(X.shape),
                                  X_corrupt)
    return X_corrupt_noise


# def image_to_tensor(train_img_dirs=["HAM10000_images_part_1"], test_img_dir="ISIC2018_Task3_Test_Images", tensor_dir="HAM10000_tensors"):
#     transform= Compose([Resize((28, 28)),
#                         ToTensor()])
    
#     train_img_list = []
#     for dir in train_img_dirs:
#         for img_name in os.listdir(dir):        # have to sort?
#             img = Image.open(dir + "/" + img_name)
#             train_img_list.append(transform(img))

#     test_img_list = []
#     for img_name in os.listdir(test_img_dir):
#         if "jpg" in img_name:
#             img = Image.open(test_img_dir + "/" + img_name)
#             test_img_list.append(transform(img))

#     if not os.path.exists(tensor_dir):
#         os.makedirs(tensor_dir)
#     torch.save(torch.stack(train_img_list), tensor_dir + "x_train.pt")
#     torch.save(torch.stack(test_img_list), tensor_dir + "x_test.pt")


# def generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file, dir="generated_data/"):
    
#     # only use train part1 (5000 data) for now
#     train_meta_data = pd.read_csv("HAM10000_metadata").sort_values(by='image_id').iloc[:5000].reset_index(drop=True)
#     x_train = torch.load("HAM10000_tensors/x_train.pt")

#     max_N = train_meta_data['dx'].value_counts().min()  # max_value for sampling

#     if N > max_N:
#         print(f"choose N s.t. N <= {max_N}")
#         raise IndexError
    
#     sampled_data = train_meta_data.groupby("dx").sample(N, random_state=123)
#     sampled_index = sampled_data.index
#     x_train_sampled = x_train[sampled_index]
#     y_train_sampled = sampled_data["dx"].map(lambda x: label_map[x])

#     test_meta_data = pd.read_csv("ISIC2018_Task3_Test_GroundTruth.csv")
#     x_test = torch.load("HAM10000_tensors/x_test.pt")
#     y_test = test_meta_data['dx'].map(lambda x: label_map[x])

#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     torch.save((y_train_sampled, y_test), dir + y_file)

#     len_cn = len(corrupt_prob_list)
#     for i in range(len_cn):
#         x_train_noise = corrupt_noise(x_train_sampled, corrupt_prob_list[i], noise_prob_list[i])
#         x_test_noise = corrupt_noise(x_test, corrupt_prob_list[i], noise_prob_list[i])
#         torch.save((x_train_noise, x_test_noise), dir + x_file_list[i])

def generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file, dir="generated_data/"):
    train_data = pd.read_csv("hmnist_28_28_RGB.csv")
    sample = train_data.groupby('label').sample(N, random_state=123)

    X = torch.from_numpy(sample.iloc[:, :-1].to_numpy())
    X = X.view(-1, 28, 28, 3).permute(0,3,1,2)
    x_train = X / 255.

    y_train = torch.from_numpy(sample.iloc[:, -1].to_numpy())

    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save((y_train, None), dir + y_file)

    len_cn = len(corrupt_prob_list)
    for i in range(len_cn):
        x_train_noise = corrupt_noise(x_train, corrupt_prob_list[i], noise_prob_list[i])
        torch.save((x_train_noise, None), dir + x_file_list[i])



if __name__ == "__main__":
    # image_to_tensor()

    # corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    corrupt_prob_list = [0.0, 0.1, 0.2, 0.3]
    noise_prob_list = [0.0, 0.1, 0.2, 0.3]
    
    len_cn = len(corrupt_prob_list)
    file_cn_list = [None] * len_cn
    for i in range(len_cn):
        file_cn_list[i] = str(int(corrupt_prob_list[i] * 100)).zfill(2) + "_" + str(int(noise_prob_list[i] * 100)).zfill(2)

    x_file_list = ["x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
    y_file = "y.pt"

    torch.manual_seed(123)
    generate_data(N, corrupt_prob_list, noise_prob_list, x_file_list, y_file)