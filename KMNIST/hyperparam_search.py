import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import json
import os
from collections import defaultdict
from pllay import AdaptiveTopoWeightLayer
from kmnist_models import ResNet18, ResNet34
from kmnist_train import train, eval


class SimpleAdaptivePllay(nn.Module):
    def __init__(self, m0, T=50, out_features=32, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.topo_layer = AdaptiveTopoWeightLayer(out_features, T, m0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_features, num_classes)

    def forward(self, input):
        x = self.flatten(input)
        x = self.topo_layer(x)
        x = self.relu(x)
        output = self.fc(x)
        return output


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # hyperparams
    # lr_list = [0.1, 0.01, 0.001]
    # batch_size_list = [32, 64, 128]
    # factor_list = [0.1, 0.3, 0.5]
    m0_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    lr = 0.01
    factor = 0.1
    # weight_decay = 0.0001
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 16
    epoch = 100
    ntimes = 5          # number of repetition for simulation of each model
    threshold = 0.005   # min value to be considered as improvement in loss
    es_patience = 4     # used for earlystopping
    sch_patience = 2    # used for lr scheduler

    torch.manual_seed(123)
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    val_dataloader = DataLoader(test_data, batch_size)
    model_list = [SimpleAdaptivePllay]

    run_name = "m0"

    # train
    for m0 in m0_list:
            
        # loop over number of simulations
        for n_sim in range(ntimes):
            print(f"\nSimulation: [{n_sim+1} / {ntimes}]")
            print("-"*30)
            
            # loop over different models
            for MODEL in model_list:                
                model = MODEL(m0).to(device)
                optim = Adam(model.parameters(), lr)
                scheduler = ReduceLROnPlateau(optim, factor=factor, patience=sch_patience, threshold=threshold)

                best_loss = float("inf")
                best_acc = None
                best_epoch = None
                early_stop_counter = 0

                writer = SummaryWriter(f"./runs/hp_search/{run_name}/{MODEL.__name__}/")

                # loop over epoch
                for n_epoch in range(epoch):
                    print(f"Model: {MODEL.__name__}")
                    print(f"Epoch: [{n_epoch+1} / {epoch}]")
                    print("-"*30)
                    train_loss, train_acc = train(model, train_dataloader, loss_fn, optim, device)
                    val_loss, val_acc = eval(model, val_dataloader, loss_fn, device)
                    
                    scheduler.step(val_loss)

                    writer.add_scalars(f"m0_{m0}/loss/sim{n_sim+1}", {"Train":train_loss, "Validation":val_loss}, n_epoch+1)
                    writer.add_scalars(f"m0_{m0}accuracy/sim{n_sim+1}", {"Train":train_acc, "Validation":val_acc}, n_epoch+1)

                    # early stopping (if loss improvement is below threshold, it's not considered as improvement)
                    if (best_loss - val_loss) > threshold:
                        early_stop_counter = 0
                        best_loss = val_loss
                        best_acc = val_acc
                        best_epoch = n_epoch
                        # torch.save(model.state_dict(), weight_dir + "/" + f"sim{n_sim+1}.pt")   # save model weights
                    else:
                        early_stop_counter += 1
                        if early_stop_counter > es_patience:    # stop training if loss doesn't improve for es_patience + 1 epochs
                            print("-"*30)
                            print(f"Epochs: [{best_epoch+1} / {epoch}]")
                            print(f"Best Validation Accuracy: {(best_acc):>0.1f}%")
                            print(f"Best Validation Loss: {best_loss:>8f}")
                            print("-"*30)
                            break
                writer.add_hparams({"m0":m0}, {"accuracy":val_acc, "loss":val_loss}, run_name=f"sim{n_sim}")
                writer.flush()                
                print("\n"*2)