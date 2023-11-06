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
from models import ResNet18
from base_models import BasePllay_05, BasePllay_2, BasePllay_05_2
import matplotlib.pyplot as plt
from generate_data import N


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

record_weight = True
record_train_info = True
record_tensorboard = True
record_grad = True

device = "cuda" if torch.cuda.is_available() else "cpu"
epoch = 100
loss_fn = nn.CrossEntropyLoss()
ntimes = 10         # number of repetition for simulation of each model
val_size = 0.3

####################################################################
run_name = "ResNet18_2layer"
####################################################################

# hyperparameters
batch_size = 32
lr = 0.001
# weight_decay = 0.0001
factor = 0.3        # factor to decay lr by when loss stagnates
threshold = 0.005   # min value to be considered as improvement in loss
es_patience = 8     # earlystopping patience
sch_patience = 3    # lr scheduler patience

corrupt_prob_list = [0.0, 0.1, 0.2, 0.3]
noise_prob_list = [0.0, 0.1, 0.2, 0.3]
# corrupt_prob_list = [0.0]
# noise_prob_list = [0.0]
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


def plot_weight_grad_norm(weight_norm, grad_norm, epoch, run_name, cn, sim):
    weight_grad_dir = f"./weight_grad/{run_name}/{cn}/{sim}"
    for (name1, weight_norm), (name2, grad_norm) in zip(weight_norm.items(), grad_norm.items()):
        plt.figure()
        plt.bar(range(1, len(weight_norm)+1), weight_norm, color='gray')
        plt.bar(range(1, len(grad_norm)+1), grad_norm, color='green')
        plt.xlabel("Nodes/Kernels")
        plt.ylabel("Weight/Grad Norm")
        plt.title(f"Epoch:{epoch}_{name1}_L1")
        plt.legend(["weight norm", "grad norm"])
        if 'gtheta' in name1:   # gtheta layer
            if not os.path.exists(weight_grad_dir + "/gtheta"):
                os.makedirs(weight_grad_dir + "/gtheta")
            plt.savefig(f"{weight_grad_dir}/gtheta/epoch{epoch}_{name1}.png")
        elif 'fc' in name1:     # fc layer
            if not os.path.exists(weight_grad_dir + "/fc"):
                os.makedirs(weight_grad_dir + "/fc")
            plt.savefig(f"{weight_grad_dir}/fc/epoch{epoch}_{name1}.png")
        else:                   # cnn layer
            if not os.path.exists(weight_grad_dir + "/conv"):
                os.makedirs(weight_grad_dir + "/conv")
            plt.savefig(f"{weight_grad_dir}/conv/epoch{epoch}_{name1}.png")
        plt.close()


def cal_weight_grad_norm(named_modules):
    weight_norm_dict = {}
    grad_norm_dict = {}
    for name, m in named_modules:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            weight, bias = list(m.parameters()) # weight shape: [out_dim, in_dim]
            if weight.requires_grad:
                weight_norm = torch.abs(weight.detach()).sum(dim=0).to("cpu")       # L1 norm
                grad_norm = torch.abs(weight.grad.detach()).sum(dim=0).to("cpu")    # L1 norm
                weight_norm_dict[name] = weight_norm
                grad_norm_dict[name] = grad_norm
    return weight_norm_dict, grad_norm_dict


def train(model, dataloader, loss_fn, optimizer, device, n_epoch, cn, sim):
    data_size = len(dataloader.dataset)
    running_avg_loss, correct, avg_signal = 0, 0, 0
    model.train()
    avg_weight_norm = {name:0 for name, m in model.named_modules() if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)) and list(m.parameters())[0].requires_grad}
    avg_grad_norm = {name:0 for name, m in model.named_modules() if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)) and list(m.parameters())[0].requires_grad}
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred, signal = model(X)
        loss = loss_fn(y_pred, y)
        running_avg_loss += (loss.item() * len(y))
        correct += (y_pred.argmax(1) == y).sum().item()
        avg_signal += signal

        loss.backward()
        optimizer.step()

        if record_grad:
            weight_norm_dict, grad_norm_dict = cal_weight_grad_norm(model.named_modules())
            avg_weight_norm = {k1:(len(y)*v1 + v2) for (k1,v1), (k2,v2) in zip(weight_norm_dict.items(), avg_weight_norm.items())}
            avg_grad_norm = {k1:(len(y)*v1 + v2) for (k1,v1), (k2,v2) in zip(grad_norm_dict.items(), avg_grad_norm.items())}
            if batch == (data_size // batch_size):  # finished one epoch
                avg_weight_norm = {k:(v/data_size) for k, v in avg_weight_norm.items()}
                avg_grad_norm = {k:(v/data_size) for k, v in avg_grad_norm.items()}
                plot_weight_grad_norm(avg_weight_norm, avg_grad_norm, n_epoch, run_name, cn, sim)   # batch_size and run_name are global params
        
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch+1) * len(y)
            print(f"Training loss: {loss:>7f} [{current:>3d}/{data_size:>3d}]")
    running_avg_loss /= data_size
    running_acc = (correct / data_size) * 100
    avg_signal /= data_size
    print(f"Train error:\n Accuracy: {(running_acc):>0.1f}%, Avg loss: {running_avg_loss:>8f} \n")
    # print(f"Average signal:\n{avg_signal}\n{avg_signal.mean().item()}\n")
    # print("-"*30)
    return running_avg_loss, running_acc


def eval(model, dataloader, loss_fn, device):
    data_size = len(dataloader.dataset)
    loss, correct, avg_signal = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred, signal = model(X)
            loss += (loss_fn(y_pred, y).item() * len(y))
            correct += (y_pred.argmax(1) == y).sum().item()
            avg_signal += signal
    loss /= data_size
    accuracy = (correct / data_size) * 100
    avg_signal /= data_size
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    # print(f"Average signal:\n{avg_signal}\n{avg_signal.mean().item()}\n")
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

            torch.manual_seed(123)               
            model = ResNet18().to(device)
            optim = Adam(model.parameters(), lr)
            scheduler = ReduceLROnPlateau(optim, factor=factor, patience=sch_patience, threshold=threshold)

            if record_train_info:
                train_info = defaultdict(list)  # used to store train info of {epoch:[...], train loss:[...], val loss:[...], train acc:[...], val acc:[...]}
            if record_tensorboard:
                writer = SummaryWriter(f"./runs/{run_name}/train/{file_cn_list[cn]}")
            if record_weight:
                weight_dir = f"./saved_weights/{run_name}/{file_cn_list[cn]}"    # directory path to store trained model weights
                if not os.path.exists(weight_dir):
                    os.makedirs(weight_dir)
            
            best_loss = float("inf")
            best_acc = None
            best_epoch = None
            early_stop_counter = 0

            # loop over epoch
            for n_epoch in range(epoch):
                print(f"Model: {ResNet18.__name__}")
                print(f"Epoch: [{n_epoch+1} / {epoch}]")
                print("-"*30)
                train_loss, train_acc = train(model, train_dataloader, loss_fn, optim, device, n_epoch+1, file_cn_list[cn], n_sim+1)
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
                if val_loss < 2.5:
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
                train_info_dir = f"./train_info/{run_name}/{file_cn_list[cn]}/"
                if not os.path.exists(train_info_dir):
                    os.makedirs(train_info_dir)
                with open(train_info_dir + "/" + f"sim{n_sim+1}_train_info.json", "w", encoding="utf-8") as f:
                    json.dump(train_info, f, indent="\t")
            
            print("\n"*2)