import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from pllay import TopoWeightLayer


class MNISTCustomDataset(Dataset):
    def __init__(self, x_dir, y_dir, train=True, random_seed=1, val_size=0.3):
        self.train = train
        x_train, x_test = torch.load(x_dir)
        y_train, y_test = torch.load(y_dir)
        self.x_tr, self.x_val, self.y_tr, self.y_val = train_test_split(x_train, y_train, test_size=val_size, shuffle=True,
                                                                        random_state=random_seed, stratify=y_train)
    
    def __len__(self):
        if self.train:
            return len(self.y_tr)
        else:
            return len(self.y_val)

    def __getitem__(self, ind):
        if self.train:
            return self.x_tr[ind], self.y_tr[ind]   # train data
        else:
            return self.x_val[ind], self.y_val[ind] # validation data


class MNISTCNN_Pi(nn.Module):
    def __init__(self, out_features=32, kernel_size=3,):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, out_features, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(out_features, 1, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Flatten())
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                          TopoWeightLayer(out_features, m0=0.05, tseq=np.linspace(0.06, 0.3, 25), K_max=2),
                                          nn.ReLU())
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                          TopoWeightLayer(out_features, m0=0.2, tseq=np.linspace(0.14, 0.4, 27), K_max=3),
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

class CNN(nn.Module):
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
    model.eval()
    loss, correct = 0, 0
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


# if __name__ == "__main__'":
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
loss_fn = nn.CrossEntropyLoss()
batch_size = 16
epoch = 100

corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
len_cn = len(corrupt_prob_list)
file_cn_list = [None] * len_cn
for i in range(len_cn):
    file_cn_list[i] = str(int(corrupt_prob_list[i] * 100)).zfill(2) + \
        '_' + str(int(noise_prob_list[i] * 100)).zfill(2)
x_dir_list = ["generated_data/mnist_x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
y_dir = "generated_data/mnist_y.pt"

ntimes = 20     # number of repetition for simulation of each model

torch.manual_seed(123)
rand_seed_list = [torch.randint(0,100, size=(1,)).item() for i in range(ntimes)]    # seed used for train_test split


for i in range(len_cn):
    print(f"Corruption rate: {corrupt_prob_list[i]}")
    print(f"Noise rate: {noise_prob_list[i]}")
    print("-"*30)    
    x_dir = x_dir_list[i]
    for n_sim in range(ntimes):
        print(f"Simulation: [{n_sim} / {ntimes}]")
        print("-"*30)
        train_dataset = MNISTCustomDataset(x_dir, y_dir, train=True, random_seed=rand_seed_list[n_sim])
        val_dataset = MNISTCustomDataset(x_dir, y_dir, train=False, random_seed=rand_seed_list[n_sim])
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

        torch.manual_seed(rand_seed_list[n_sim])
        model = MNISTCNN_Pi().to(device)
        optim = Adam(model.parameters(), lr)
        # scheduler and earlystopping
        min_loss_1 = 100
        for n_epoch in range(epoch):
            print(f'Model: {model.__class__}')
            print(f"Epoch: [{n_epoch} / {epoch}]")
            print("-"*30)
            train_loss = train(model, train_dataloader, loss_fn, optim, device)
            val_loss, val_acc = eval(model, val_dataloader, loss_fn, device)
            if val_loss < min_loss_1:
                min_loss_1 = val_loss
                torch.save(model.state_dict(), './pllayCNNweight' + corrupt_prob_list[i] + '_' + noise_prob_list[i] + '.pt')

        # baseline
        torch.manual_seed(rand_seed_list[n_sim])
        model = MNISTCNN_Pi().to(device)
        optim = Adam(model.parameters(), lr)
        min_loss_2 = 100
        for n_epoch in range(epoch):
            print(f'Model: {model.__class__}')
            print(f"Epoch: [{n_epoch} / {epoch}]")
            print("-"*30)
            train_loss = train(model, train_dataloader, loss_fn, optim, device)
            val_loss, val_acc = eval(model, val_dataloader, loss_fn, device)
            if val_loss < min_loss_2:
                min_loss_2 = val_loss
                torch.save(model.state_dict(), './CNNweight' + corrupt_prob_list[i] + '_' + noise_prob_list[i] + '.pt')