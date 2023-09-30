import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from pllay import TopoWeightLayer
import os


class MnistCustomDataset(Dataset):
    def __init__(self, x_dir, y_dir, mode="train", random_seed=1, val_size=0.3):
        """
        Args:
            mode: one of "train", "val", or "test"
        """
        assert mode in ("train", "val", "test")
        self.mode = mode
        x_train, self.x_test = torch.load(x_dir)
        y_train, self.y_test = torch.load(y_dir)
        if self.mode == "train" or self.mode == "val":
            self.x_tr, self.x_val, self.y_tr, self.y_val = train_test_split(x_train, y_train, test_size=val_size, shuffle=True,
                                                                            random_state=random_seed, stratify=y_train)
    
    def __len__(self):
        if self.mode == "train":
            return len(self.y_tr)
        elif self.mode == "val":
            return len(self.y_val)
        else:
            return len(self.y_test)

    def __getitem__(self, ind):
        if self.mode == "train":
            return self.x_tr[ind], self.y_tr[ind]       # train data
        elif self.mode == "val":
            return self.x_val[ind], self.y_val[ind]     # validation data
        else:
            return self.x_test[ind], self.y_test[ind]   # test data


class MnistCnn_Pi(nn.Module):
    def __init__(self, out_features=32, kernel_size=3,):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, out_features, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(out_features, 1, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Flatten())
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                          TopoWeightLayer(out_features, tseq=np.linspace(0.06, 0.3, 25), m0=0.05, K_max=2),
                                          nn.ReLU())
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                          TopoWeightLayer(out_features, tseq=np.linspace(0.14, 0.4, 27), m0=0.2, K_max=3),
                                          nn.ReLU())
        self.linear_layer = nn.Sequential(nn.Linear(784+out_features+out_features, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 10),
                                          nn.Softmax(-1))

    def forward(self, input):
        x = self.conv_layer(input)
        x_1 = self.topo_layer_1(input)
        x_2 = self.topo_layer_2(input)
        x_3 = torch.concat((x, x_1, x_2), dim=-1)
        out = self.linear_layer(x_3)
        return out


class MnistCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1,32,kernel_size=3,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(32,1,kernel_size=3,padding=1),
                                        nn.ReLU(),
                                        nn.Flatten())
        self.linear_layer = nn.Sequential(nn.Linear(784, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 10),
                                            nn.Softmax(-1))
    
    def forward(self, input):
        x = self.conv_layer(input)
        out = self.linear_layer(x)
        return out


def train(model, dataloader, loss_fn, optimizer, device):
    data_size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch+1) * len(y)
            print(f"Training loss: {loss:>7f} [{current:>3d}/{data_size:>3d}]")
    return loss


def eval(model, dataloader, loss_fn, device):
    data_size = len(dataloader.dataset)
    loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += (loss_fn(y_pred, y).item() * len(y))
            correct += (y_pred.argmax(1) == y).sum().item()
    loss /= data_size
    correct /= data_size
    accuracy = correct * 100
    print(f"Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    return loss, accuracy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.001
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 16
    epoch = 100
    ntimes = 20     # number of repetition for simulation of each model
    min_delta = 0.003   # min value to be considered as improvement in loss
    patience = 3        # used for earlystopping

    corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    len_cn = len(corrupt_prob_list)
    file_cn_list = [None] * len_cn
    for cn in range(len_cn):
        file_cn_list[cn] = str(int(corrupt_prob_list[cn] * 100)).zfill(2) + "_" + str(int(noise_prob_list[cn] * 100)).zfill(2)
    x_dir_list = ["./generated_data/mnist_x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
    y_dir = "./generated_data/mnist_y.pt"

    torch.manual_seed(123)
    rand_seed_list = [torch.randint(0,100, size=(1,)).item() for i in range(ntimes)]    # seed used for train_test split

    if not os.path.exists("./checkpoints_pCNN/"):
        os.makedirs("./checkpoints_pCNN/")

    if not os.path.exists("./checkpoints_CNN/"):
        os.makedirs("./checkpoints_CNN/")

    for cn in range(len_cn):
        print(f"Corruption rate: {corrupt_prob_list[cn]}")
        print(f"Noise rate: {noise_prob_list[cn]}")
        print("-"*30)    
        x_dir = x_dir_list[cn]
        for n_sim in range(ntimes):
            print(f"Simulation: [{n_sim+1} / {ntimes}]")
            print("-"*30)
            train_dataset = MnistCustomDataset(x_dir, y_dir, mode="train", random_seed=rand_seed_list[n_sim])
            val_dataset = MnistCustomDataset(x_dir, y_dir, mode="val", random_seed=rand_seed_list[n_sim])
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

            # train CNN + Pllay(i)
            if not os.path.exists(f"./checkpoints_pCNN/sim{n_sim+1}/"):
                os.makedirs(f"./checkpoints_pCNN/sim{n_sim+1}/")
            torch.manual_seed(rand_seed_list[n_sim])
            model = MnistCnn_Pi().to(device)
            optim = Adam(model.parameters(), lr)
            best_loss_pllay = float("inf")
            early_stop_counter_pllay = 0
            for n_epoch in range(epoch):
                print(f"Model: {model.__class__}")
                print(f"Epoch: [{n_epoch+1} / {epoch}]")
                print("-"*30)
                train_loss = train(model, train_dataloader, loss_fn, optim, device)
                val_loss, val_acc = eval(model, val_dataloader, loss_fn, device)
                
                # early stopping 
                if val_loss < best_loss_pllay:
                    torch.save(model.state_dict(), f"./checkpoints_pCNN/sim{n_sim+1}/pCNNweight" + file_cn_list[cn] + ".pt")
                    if (best_loss_pllay - val_loss) < min_delta:  # if loss improvement is below min_delta, it's not considered as improvement
                        early_stop_counter_cnn += 1
                        if early_stop_counter_cnn == patience:
                            print("-"*30)
                            print("Early Stopping at Epoch:", n_epoch)
                            print("Best Validation Loss: ", val_loss)
                            print("-"*30)
                            break
                    else:
                        early_stop_counter_cnn = 0
                    best_loss_pllay = val_loss
                else:
                    early_stop_counter_cnn += 1
                    if early_stop_counter_cnn == patience:
                        print("-"*30)
                        print("Early Stopping at Epoch:", n_epoch)
                        print("Best Validation Loss: ", best_loss_pllay)
                        print("-"*30)
                        break


            # train baseline CNN
            if not os.path.exists(f"./checkpoints_CNN/sim{n_sim+1}/"):
                os.makedirs(f"./checkpoints_CNN/sim{n_sim+1}/")
            torch.manual_seed(rand_seed_list[n_sim])
            model = MnistCnn.to(device)
            optim = Adam(model.parameters(), lr)
            best_loss_cnn = float("inf")
            early_stop_counter_cnn = 0
            for n_epoch in range(epoch):
                print(f"Model: {model.__class__}")
                print(f"Epoch: [{n_epoch} / {epoch}]")
                print("-"*30)
                train_loss = train(model, train_dataloader, loss_fn, optim, device)
                val_loss, val_acc = eval(model, val_dataloader, loss_fn, device)
                
                # early stopping 
                if val_loss < best_loss_cnn:
                    torch.save(model.state_dict(), f"./checkpoints_CNN/sim{n_sim+1}/CNNweight" + file_cn_list[cn] + ".pt")
                    if (best_loss_cnn - val_loss) < min_delta:  # if loss improvement is below min_delta, it's not considered as improvement
                        early_stop_counter_cnn += 1
                        if early_stop_counter_cnn == patience:
                            print("-"*30)
                            print("Early Stopping at Epoch:", n_epoch)
                            print("Best Validation Loss: ", val_loss)
                            print("-"*30)
                            break
                    else:
                        early_stop_counter_cnn = 0
                    best_loss_cnn = val_loss
                else:
                    early_stop_counter_cnn += 1
                    if early_stop_counter_cnn == patience:
                        print("-"*30)
                        print("Early Stopping at Epoch:", n_epoch)
                        print("Best Validation Loss: ", best_loss_cnn)
                        print("-"*30)
                        break