import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import json
import os
from collections import defaultdict
from models import ResNet18, AdPRNet18
from base_models import Pllay
import matplotlib.pyplot as plt
import re


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

record_weight = True
record_train_info = False
record_tensorboard = False

# model_list = [BaseAdPllay]
model_list = [AdPRNet18]
# model_list = [ResNet18]
device = "cuda" if torch.cuda.is_available() else "cpu"
epoch = 100
loss_fn = nn.CrossEntropyLoss()
ntimes = 1         # number of repetition for simulation of each model
val_size = 0.3
run_name = f"t{int((1-val_size)*1000)}" + "_".join([model.__name__ for model in model_list])

# hyperparameters
batch_size = 32
lr = 0.001
# lr = 0.03
# weight_decay = 0.0001
factor = 0.1        # factor to decay lr by when loss stagnates
threshold = 0.005   # min value to be considered as improvement in loss
es_patience = 4     # earlystopping patience
sch_patience = 2    # lr scheduler patience

# corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
# noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
# corrupt_prob_list = [0.0, 0.1, 0.2, 0.3]
# noise_prob_list = [0.0, 0.1, 0.2, 0.3]
corrupt_prob_list = [0.0]
noise_prob_list = [0.0]
len_cn = len(corrupt_prob_list)
file_cn_list = [None] * len_cn
for cn in range(len_cn):
    file_cn_list[cn] = str(int(corrupt_prob_list[cn] * 100)).zfill(2) + "_" + str(int(noise_prob_list[cn] * 100)).zfill(2)
x_dir_list = ["./generated_data/x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
y_dir = "./generated_data/y.pt"


class CustomDataset(Dataset):
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


def plot_weight_grad(named_params, epoch, run_name):
    pattern = re.compile("1.weight")    # pattern to filter batchnorm2d
    weight_grad_dir = f"./weight_grad/{run_name}"
    if not os.path.exists(weight_grad_dir):
        os.makedirs(weight_grad_dir)
    for name, weight in named_params:
        if ("bias" not in name) and ("downsample" not in name) and not pattern.search(name):
            if "conv" in name:  # conv2d layers
                pass
            elif "avg" in name: # landscape weighted avg. layer
                pass
            else:               # fc layers
                weight_norm = (weight.detach() ** 2).sum(dim=0).sqrt().to("cpu")
                grad_norm = (weight.grad.detach() ** 2).sum(dim=0).sqrt().to("cpu")
                plt.figure()
                plt.bar(range(1, len(weight_norm)+1), weight_norm, color='gray')
                plt.bar(range(1, len(grad_norm)+1), grad_norm, color='green')
                if "fc" in name:
                    plt.vlines(256, -0.05, 0.05, colors='red')
                plt.xlabel("Nodes")
                plt.ylabel("Weight/Grad Norm")
                plt.title(f"Epoch:{epoch}_{name}")
                plt.legend(['boundary', 'weight', 'grad'])
                plt.savefig(f"{weight_grad_dir}/epoch{epoch}_{name}.png")


def train(model, dataloader, loss_fn, optimizer, device, n_epoch):
    data_size = len(dataloader.dataset)
    running_avg_loss, correct = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        running_avg_loss += (loss.item() * len(y))
        correct += (y_pred.argmax(1) == y).sum().item()

        loss.backward()

        optimizer.step()
        if batch == (data_size // batch_size):  # finished one epoch
            plot_weight_grad(model.named_parameters(), n_epoch, run_name)   # batch_size and run_name are global params
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch+1) * len(y)
            print(f"Training loss: {loss:>7f} [{current:>3d}/{data_size:>3d}]")
    running_avg_loss /= data_size
    running_acc = (correct / data_size) * 100
    print(f"Train error:\n Accuracy: {(running_acc):>0.1f}%, Avg loss: {running_avg_loss:>8f} \n")
    return running_avg_loss, running_acc


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
    accuracy = (correct / data_size) * 100
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    return loss, accuracy


if __name__ == "__main__":
    torch.manual_seed(123)
    rand_seed_list = [torch.randint(0,100, size=(1,)).item() for i in range(ntimes)]    # used to create different train/val split for each simulation

    # train
    # loop over data with different corruption/noise probability
    for cn in range(len_cn):
        print("-"*30)
        print(f"Corruption/Noise rate: {file_cn_list[cn]}")
        print("-"*30)
        
        # loop over number of simulations
        for n_sim in range(ntimes):
            print(f"\nSimulation: [{n_sim+1} / {ntimes}]")
            print("-"*30)
            train_dataset = CustomDataset(x_dir_list[cn], y_dir, mode="train", random_seed=rand_seed_list[n_sim], val_size=val_size)
            val_dataset = CustomDataset(x_dir_list[cn], y_dir, mode="val", random_seed=rand_seed_list[n_sim], val_size=val_size)
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
            
            # loop over different models
            for MODEL in model_list:
                torch.manual_seed(123)                
                model = MODEL().to(device)
                # if isinstance(model, AdaptivePRNet18):
                #     pl_pre_dir = f"saved_weights/t0.7_v0.3_BaseAdPllay_BaseAdPllay_not_robust_BaseAdPllay_no_t_BaseAdPllay_no_t_not_robust/x_{file_cn_list[cn]}/BaseAdPllay/sim{n_sim+1}.pt"
                #     model.load_pretrained_weights(pl_pre_dir, rn_pre_dir)
                # optim = Adam(model.parameters(), lr, weight_decay=weight_decay)
                optim = Adam(model.parameters(), lr)
                scheduler = ReduceLROnPlateau(optim, factor=factor, patience=sch_patience, threshold=threshold)

                if record_train_info:
                    train_info = defaultdict(list)  # used to store train info of {epoch:[...], train loss:[...], val loss:[...], train acc:[...], val acc:[...]}
                if record_tensorboard:
                    writer = SummaryWriter(f"./runs/{run_name}/train/{file_cn_list[cn]}/{MODEL.__name__}")
                if record_weight:
                    weight_dir = f"./saved_weights/{run_name}/{file_cn_list[cn]}/{MODEL.__name__}"    # directory path to store trained model weights
                    if not os.path.exists(weight_dir):
                        os.makedirs(weight_dir)
                
                best_loss = float("inf")
                best_acc = None
                best_epoch = None
                early_stop_counter = 0

                # loop over epoch
                for n_epoch in range(epoch):
                    print(f"Model: {MODEL.__name__}")
                    print(f"Epoch: [{n_epoch+1} / {epoch}]")
                    print("-"*30)
                    train_loss, train_acc = train(model, train_dataloader, loss_fn, optim, device, n_epoch+1)
                    val_loss, val_acc = eval(model, val_dataloader, loss_fn, device)
                    
                    scheduler.step(val_loss)

                    if record_train_info:
                        # save train information
                        train_info['epoch'].append(n_epoch+1)
                        train_info['train/val loss'].append((train_loss, val_loss))
                        train_info['train/val acc'].append((train_acc, val_acc))
                    if record_tensorboard:
                        # write to tensorboard
                        writer.add_scalars(f"loss/sim{n_sim+1}", {"Train":train_loss, "Validation":val_loss}, n_epoch+1)
                        writer.add_scalars(f"accuracy/sim{n_sim+1}", {"Train":train_acc, "Validation":val_acc}, n_epoch+1)
                        writer.flush()
                    
                    # early stopping (if loss improvement is below threshold, it's not considered as improvement)
                    if (best_loss - val_loss) > threshold:
                        early_stop_counter = 0
                        best_loss = val_loss
                        best_acc = val_acc
                        best_epoch = n_epoch
                        if record_weight:
                            torch.save(model.state_dict(), weight_dir + "/" + f"sim{n_sim+1}.pt")   # save model weights
                    else:
                        early_stop_counter += 1
                        if early_stop_counter > es_patience:    # stop training if loss doesn't improve for es_patience + 1 epochs
                            print("-"*30)
                            print(f"Epochs: [{best_epoch+1} / {epoch}]")
                            print(f"Best Validation Accuracy: {(best_acc):>0.1f}%")
                            print(f"Best Validation Loss: {best_loss:>8f}")
                            print("-"*30)
                            break
                
                if record_train_info:
                    # save train info as json file
                    train_info_dir = f"./train_info/{run_name}/{file_cn_list[cn]}/{MODEL.__name__}"
                    if not os.path.exists(train_info_dir):
                        os.makedirs(train_info_dir)
                    with open(train_info_dir + "/" + f"sim{n_sim+1}_train_info.json", "w", encoding="utf-8") as f:
                        json.dump(train_info, f, indent="\t")
                
                print("\n"*2)