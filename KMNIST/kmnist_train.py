import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from pllay import TopoWeightLayer
import json
import os


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


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU())
        self.conv_layer2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()


    def forward(self, input):
        x = self.conv_layer1(input)
        x = self.conv_layer2(x)
        x = x + input if self.downsample is None else x + self.downsample(input)
        output = self.relu(x)
        return output


class ResNet(nn.Module):
    def __init__(self, block, cfg, num_classes=10):
        super().__init__()
        # change architecture of ResNet bc. our image size (3, 32, 32) is too small for the original architecture
        # self.in_channels = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.bn = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()

        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 32   # channel of input that goes into res_layer1

        self.conv_layer = nn.Sequential(nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.in_channels),
                                        nn.ReLU())
        
        self.res_layer_1 = self._make_layers(block, 64, cfg[0], stride=1)
        self.res_layer_2 = self._make_layers(block, 128, cfg[1], stride=2)
        self.res_layer_3 = self._make_layers(block, 256, cfg[2], stride=2)
        self.res_layer_4 = self._make_layers(block, 512, cfg[3], stride=2)

        self.fc_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(512 * block.expansion, num_classes))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, first_conv_channel, num_blocks, stride):      
        if stride != 1 or self.in_channels != first_conv_channel * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, first_conv_channel*block.expansion, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(first_conv_channel, block.expansion))
        else:
            downsample = None
        
        block_list =[]
        block_list.append(block(self.in_channels, first_conv_channel, stride, downsample))

        self.in_channels = first_conv_channel * block.expansion
        
        for _ in range(1, num_blocks):
            block_list.append(block(self.in_channels, first_conv_channel))
        return nn.Sequential(*block_list)

    def forward(self, input):
        x = self.conv_layer(input)
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        output = self.fc_layer(x)
        return output


class PllayResNet(nn.Module):
    def __init__(self, block, cfg, out_features=32, num_classes=10):
        super().__init__()
        self.in_channels = 32   # channel of input that goes into res_layer1

        self.conv_layer = nn.Sequential(nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.in_channels),
                                        nn.ReLU())
        
        self.res_layer_1 = self._make_layers(block, 64, cfg[0], stride=1)
        self.res_layer_2 = self._make_layers(block, 128, cfg[1], stride=2)
        self.res_layer_3 = self._make_layers(block, 256, cfg[2], stride=2)
        self.res_layer_4 = self._make_layers(block, 512, cfg[3], stride=2)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten())

        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        TopoWeightLayer(out_features, tseq=np.linspace(0.06, 0.3, 25), m0=0.05, K_max=2))    # hyperparameter 수정
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        TopoWeightLayer(out_features, tseq=np.linspace(0.14, 0.4, 27), m0=0.2, K_max=3))     # hyperparameter 수정
        
        self.fc = nn.Linear(512*block.expansion + 2*out_features, num_classes)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, first_conv_channel, num_blocks, stride):      
        if stride != 1 or self.in_channels != first_conv_channel * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, first_conv_channel*block.expansion, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(first_conv_channel, block.expansion))
        else:
            downsample = None
        
        block_list =[]
        block_list.append(block(self.in_channels, first_conv_channel, stride, downsample))

        self.in_channels = first_conv_channel * block.expansion
        
        for _ in range(1, num_blocks):
            block_list.append(block(self.in_channels, first_conv_channel))
        return nn.Sequential(*block_list)
    
    def forward(self, input):
        x = self.conv_layer(input)
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x_1 = self.pool(x)

        x_2 = self.topo_layer_1(input)
        x_3 = self.topo_layer_2(input)

        output = self.fc(torch.concat((x_1, x_2, x_3), dim=-1))
        return output


class ResNet18(ResNet):
    def __init__(self, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10):
        super().__init__(block, cfg, num_classes)


class ResNet34(ResNet):
    def __init__(self, block=ResidualBlock, cfg=[3,4,6,3], num_classes=10):
        super().__init__(block, cfg, num_classes)


class PllayResNet18(PllayResNet):
    def __init__(self, block=ResidualBlock, cfg=[2,2,2,2], out_features=32, num_classes=10):
        super().__init__(block, cfg, out_features, num_classes)


class PllayResNet34(PllayResNet):
    def __init__(self, block=ResidualBlock, cfg=[3,4,6,3], out_features=32, num_classes=10):
        super().__init__(block, cfg, out_features, num_classes)


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
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    return loss, accuracy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.1
    weight_decay = 0.00001
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 16
    epoch = 100
    ntimes = 20         # number of repetition for simulation of each model
    min_delta = 0.003   # min value to be considered as improvement in loss
    patience = 3        # used for earlystopping

    # corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    corrupt_prob_list = [0.0, 0.15]
    noise_prob_list = [0.0, 0.15]

    len_cn = len(corrupt_prob_list)
    file_cn_list = [None] * len_cn
    for cn in range(len_cn):
        file_cn_list[cn] = str(int(corrupt_prob_list[cn] * 100)).zfill(2) + "_" + str(int(noise_prob_list[cn] * 100)).zfill(2)
    x_dir_list = ["./generated_data/mnist_x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
    y_dir = "./generated_data/mnist_y.pt"

    torch.manual_seed(123)
    rand_seed_list = [torch.randint(0,100, size=(1,)).item() for i in range(ntimes)]    # used to create different train/val split for each simulation
    model_list = [ResNet18(), PllayResNet18(), ResNet34(), PllayResNet34()]

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
            for model in model_list:
                train_info = []    # list to store train info of (epoch, accuracy, loss)}

                weight_dir = f"./saved_weights/mnist_x_{file_cn_list[cn]}/{model._get_name()}"     # directory path to store trained model weights
                if not os.path.exists(weight_dir):
                    os.makedirs(weight_dir)
                
                model = model.to(device)
                optim = Adam(model.parameters(), lr, weight_decay=weight_decay)     # no lr scheduler for now
                best_loss = float("inf")
                best_acc = None
                best_epoch = None
                early_stop_counter = 0
                
                # loop over epoch
                for n_epoch in range(epoch):
                    print(f"Model: {model._get_name()}")
                    print(f"Epoch: [{n_epoch+1} / {epoch}]")
                    print("-"*30)
                    train(model, train_dataloader, loss_fn, optim, device)
                    val_loss, val_acc = eval(model, val_dataloader, loss_fn, device)
                    
                    train_info.append((n_epoch+1, val_acc, val_loss))

                    # early stopping (if loss improvement is below min_delta, it's not considered as improvement)
                    if (best_loss - val_loss) >= min_delta:
                        early_stop_counter = 0
                        best_loss = val_loss
                        best_acc = val_acc
                        best_epoch = n_epoch
                        torch.save(model.state_dict(), weight_dir + "/" + f"sim{n_sim+1}.pt")
                    else:
                        early_stop_counter += 1
                        if early_stop_counter == patience:
                            train_info = train_info[:(best_epoch+1)]
                            print("-"*30)
                            print(f"Epochs: [{best_epoch+1} / {epoch}]")
                            print(f"Best Validation Accuracy: {(best_acc):>0.1f}%")
                            print(f"Best Validation Loss: {best_loss:>8f}")
                            print("-"*30)
                            break
                print("\n"*2)
        
                # save train info as json file
                train_info_dir = f"./train_info/mnist_x_{file_cn_list[cn]}/{model._get_name()}"
                if not os.path.exists(train_info_dir):
                    os.makedirs(train_info_dir)
                with open(train_info_dir + "/" + f"sim{n_sim+1}_train_info.json", "w", encoding="utf-8") as f:
                    json.dump(train_info, f)