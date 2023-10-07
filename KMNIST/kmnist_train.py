import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
from collections import defaultdict
from kmnist_models import ResNet18, ResNet34


class KMNISTCustomDataset(Dataset):
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


def eval(model, dataloader, loss_fn, device, mode="Validation"):
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
    print(f"{mode} error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    return loss, accuracy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.01
    weight_decay = 0.0001
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32
    epoch = 100
    ntimes = 20         # number of repetition for simulation of each model
    threshold = 0.005   # min value to be considered as improvement in loss
    es_patience = 4     # used for earlystopping
    sch_patience = 2    # used for lr scheduler

    # corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    corrupt_prob_list = [0.0, 0.15]
    noise_prob_list = [0.0, 0.15]

    len_cn = len(corrupt_prob_list)
    file_cn_list = [None] * len_cn
    for cn in range(len_cn):
        file_cn_list[cn] = str(int(corrupt_prob_list[cn] * 100)).zfill(2) + "_" + str(int(noise_prob_list[cn] * 100)).zfill(2)
    x_dir_list = ["./generated_data/x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
    y_dir = "./generated_data/y.pt"

    torch.manual_seed(123)
    rand_seed_list = [torch.randint(0,100, size=(1,)).item() for i in range(ntimes)]    # used to create different train/val split for each simulation
    # model_list = [ResNet18(), PllayResNet18(), ResNet34(), PllayResNet34()]
    model_list = [ResNet18, ResNet34]

    # train
    # loop over data with different corruption/noise probability
    for cn in range(len_cn):
        print("-"*30)
        print(f"Corruption rate: {corrupt_prob_list[cn]}")
        print(f"Noise rate: {noise_prob_list[cn]}")
        print("-"*30)
        
        # loop over number of simulations
        for n_sim in range(ntimes):
            print(f"\nSimulation: [{n_sim+1} / {ntimes}]")
            print("-"*30)
            train_dataset = KMNISTCustomDataset(x_dir_list[cn], y_dir, mode="train", random_seed=rand_seed_list[n_sim])
            val_dataset = KMNISTCustomDataset(x_dir_list[cn], y_dir, mode="val", random_seed=rand_seed_list[n_sim])
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
            
            # loop over different models
            for MODEL in model_list:                
                model = MODEL().to(device)
                optim = Adam(model.parameters(), lr, weight_decay=weight_decay)
                scheduler = ReduceLROnPlateau(optim, factor=0.1, patience=sch_patience, threshold=threshold)

                weight_dir = f"./saved_weights/x_{file_cn_list[cn]}/{model._get_name()}"    # directory path to store trained model weights
                if not os.path.exists(weight_dir):
                    os.makedirs(weight_dir)

                best_loss = float("inf")
                best_acc = None
                best_epoch = None
                early_stop_counter = 0
                train_info = defaultdict(list)  # used to store train info of {epoch:[...], train loss:[...], val loss:[...], train acc:[...], val acc:[...]}
                
                # loop over epoch
                for n_epoch in range(epoch):
                    print(f"Model: {model._get_name()}")
                    print(f"Epoch: [{n_epoch+1} / {epoch}]")
                    print("-"*30)
                    train(model, train_dataloader, loss_fn, optim, device)
                    train_loss, train_acc = eval(model, train_dataloader, loss_fn, device, mode="Train")
                    val_loss, val_acc = eval(model, val_dataloader, loss_fn, device, mode="Validation")
                    
                    scheduler.step(val_loss)

                    train_info['epoch'].append(n_epoch+1)
                    train_info['train loss'].append(train_loss)
                    train_info['val loss'].append(val_loss)
                    train_info['train acc'].append(train_acc)
                    train_info['val acc'].append(val_acc)

                    # early stopping (if loss improvement is below threshold, it's not considered as improvement)
                    if (best_loss - val_loss) > threshold:
                        early_stop_counter = 0
                        best_loss = val_loss
                        best_acc = val_acc
                        best_epoch = n_epoch
                        torch.save(model.state_dict(), weight_dir + "/" + f"sim{n_sim+1}.pt")
                    else:
                        early_stop_counter += 1
                        if early_stop_counter > es_patience:
                            print("-"*30)
                            print(f"Epochs: [{best_epoch+1} / {epoch}]")
                            print(f"Best Validation Accuracy: {(best_acc):>0.1f}%")
                            print(f"Best Validation Loss: {best_loss:>8f}")
                            print("-"*30)
                            break
                print("\n"*2)
        
                # save train info as json file
                train_info_dir = f"./train_info/x_{file_cn_list[cn]}/{model._get_name()}"
                if not os.path.exists(train_info_dir):
                    os.makedirs(train_info_dir)
                with open(train_info_dir + "/" + f"sim{n_sim+1}_train_info.json", "w", encoding="utf-8") as f:
                    json.dump(train_info, f, indent="\t")