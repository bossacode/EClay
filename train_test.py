import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
from models import EClayResNet, EC_CNN_2, PL_CNN_2
import os
import wandb


class CustomDataset(Dataset):
    def __init__(self, x_path, y_path, mode, val_size=0.3, seed=123):
        """
        Args:
            mode: one of "train", "val", or "test"
        """
        assert mode in ("train", "val", "test")
        self.mode = mode

        x_train, self.x_test = torch.load(x_path)
        y_train, self.y_test = torch.load(y_path)
        if self.mode == "train" or self.mode == "val":
            self.x_tr, self.x_val, self.y_tr, self.y_val = train_test_split(x_train, y_train, test_size=val_size, shuffle=True,
                                                                            random_state=seed, stratify=y_train)
    
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


class EarlyStopping:
    def __init__(self, patience, threshold, val_metric="loss"):
        """
        patience:
        threshold:
        val_metric: Validation metric used to measure model improvement. One of "loss" or "acc".
        """
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.best_loss, self.best_acc, self.best_epoch = float("inf"), 0, None
        self.val_metic = val_metric

    def stop_training(self, val_loss, val_acc, epoch):
        stop, improvement = True, True
        diff = (self.best_loss - val_loss) if self.val_metic == "loss" else (val_acc - self.best_acc)
        if diff > self.threshold:   # improvement needs to be above threshold 
            self.count = 0
            self.best_loss, self.best_acc, self.best_epoch = val_loss, val_acc, epoch
            return not stop, improvement
        else:
            self.count += 1
            if self.count > self.patience:  # stop training if no improvement for patience + 1 epochs
                print("-"*30)
                print(f"Best Epoch: {self.best_epoch}")
                print(f"Best Validation Accuracy: {(self.best_acc):>0.1f}%")
                print(f"Best Validation Loss: {self.best_loss:>8f}")
                print("-"*30)
                return stop, not improvement
            return not stop, not improvement


def train(model, dataloader, loss_fn, optimizer, device):
    """
    train for 1 epoch
    """
    data_size = len(dataloader.dataset)
    ma_loss, correct = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader, 1):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        ma_loss += (loss.item() * len(y))  # bc. loss_fn predicts avg loss
        correct += (y_pred.argmax(1) == y).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 1:
            print(f"Training loss: {loss.item():>7f} [{batch*len(y):>3d}/{data_size:>3d}]")
    ma_loss /= data_size                    # moving average of loss over 1 epoch
    ma_acc = (correct / data_size) * 100    # moving average of accuracy over 1 epoch
    print(f"Train error:\n Accuracy: {ma_acc:>0.1f}%, Avg loss: {ma_loss:>8f} \n")
    return ma_loss, ma_acc


def eval(model, dataloader, loss_fn, device):
    """
    """
    y_pred_list = []
    data_size = len(dataloader.dataset)
    loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += (loss_fn(y_pred, y).item() * len(y))
            correct += (y_pred.argmax(1) == y).sum().item()
            y_pred_list.append(y_pred.argmax(1))
    loss /= data_size
    accuracy = (correct / data_size) * 100
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")

    predicted = torch.concat(y_pred_list).detach().to("cpu")
    _, ground_truth = dataloader.dataset[:]
    report = classification_report(ground_truth, predicted, zero_division="warn")
    print(report)
    return loss, accuracy


def train_val(MODEL, config, x_path, y_path, seed, weight_path=None, log_metric=False, log_grad=False, val_metric="loss"):
    """
    Args:
        MODEL:
        config:
        x_path: file path to data
        y_path: file path to label
        seed:
        weight_path:
        log: whether to log metrics to wandb
    """
    train_dataset = CustomDataset(x_path, y_path, mode="train", seed=seed, val_size=config["val_size"])
    val_dataset = CustomDataset(x_path, y_path, mode="val", seed=seed, val_size=config["val_size"])
    train_dataloader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, config["batch_size"])

    # set seed for initialization?
    model = MODEL(**config["model_params"]).to(config["device"])
    loss_fn = nn.CrossEntropyLoss()
    if isinstance(model, EClayResNet):
        optim = Adam([
            {"params": model.res_layers.parameters(), "lr": config["lr_res"]},
            {"params": model.topo_layer_1.parameters()},
            {"params": model.topo_layer_2.parameters()},
            {"params": model.fc.parameters(), "lr": config["lr_fc"]}
        ],
        lr=config["lr_topo"], weight_decay=0)
    elif isinstance(model, EC_CNN_2) or isinstance(model, PL_CNN_2):
        optim = Adam([
            {"params": model.conv_layer.parameters(), "lr": config["lr_conv"]},
            {"params": model.topo_layer_1.parameters()},
            {"params": model.topo_layer_2.parameters()},
            {"params": model.fc.parameters(), "lr": config["lr_fc"]}
        ],
        lr=config["lr_topo"], weight_decay=0)
    else:
        optim = Adam(model.parameters(), lr=config["lr"], weight_decay=0)
    
    scheduler = ReduceLROnPlateau(optim, mode="min" if val_metric == "loss" else "max",
                                  factor=config["factor"], patience=config["sch_patience"], threshold=config["threshold"], verbose=True)
    
    # set early stopping patience as 2.5 times that of scheduler patience
    es = EarlyStopping(int(config["sch_patience"] * 2.5), config["threshold"], val_metric=val_metric)

    if log_grad:
        wandb.watch(model, loss_fn, log="all", log_freq=5)  # log gradients and model parameters every 5 batches

    # train
    for n_epoch in range(1, config["epochs"]+1):
        print(f"\nEpoch: [{n_epoch} / {config['epochs']}]")
        print("-"*30)

        train_loss, train_acc = train(model, train_dataloader, loss_fn, optim, config["device"])
        val_loss, val_acc = eval(model, val_dataloader, loss_fn, config["device"])

        scheduler.step(val_loss if val_metric == "loss" else val_acc)

        # early stopping
        stop, improvement = es.stop_training(val_loss, val_acc, n_epoch)
        if log_metric:
            wandb.log({"train":{"loss":train_loss, "accuracy":train_acc},
                    "val":{"loss":val_loss, "accuracy":val_acc},
                    "best_val":{"loss":es.best_loss, "accuracy":es.best_acc}}, step=n_epoch)
        if stop:
            if weight_path is not None:
                torch.save(model_state_dict, weight_path)   # save model weights
            break
        elif improvement and weight_path is not None:
            model_state_dict = model.state_dict()


def train_test(MODEL, config, x_path, y_path, seed, weight_path, log_metric=False, log_grad=False, val_metric="loss"):
    """
    Args:
        MODEL:
        config:
        x_path: file path to data
        y_path: file path to label
        seed:
        weight_path:
        log: whether to log metrics to wandb
    """
    train_val(MODEL, config, x_path, y_path, seed, weight_path, log_metric, log_grad, val_metric)

    test_dataset = CustomDataset(x_path, y_path, mode="test")
    test_dataloader = DataLoader(test_dataset, config["batch_size"])
    loss_fn = nn.CrossEntropyLoss()
    
    # test
    model = MODEL(**config["model_params"]).to(config["device"])
    model.load_state_dict(torch.load(weight_path, map_location=config["device"]))
    test_loss, test_acc = eval(model, test_dataloader, loss_fn, config["device"])
    if log_metric: wandb.log({"test":{"loss":test_loss, "accuracy":test_acc}})


def train_val_wandb(MODEL, config, x_path, y_path, seed, weight_path=None, log_metric=True, log_grad=False, project=None, group=None, job_type=None, val_metric="loss"):
    """
    Args:
        MODEL:
        config:
        x_path: file path to data
        y_path: file path to label
        seed:
        weight_path:
        log: whether to log metrics to wandb
    """
    with wandb.init(config=config, project=project, group=group, job_type=job_type):
        config = wandb.config
        train_val(MODEL, config, x_path, y_path, seed, weight_path, log_metric, log_grad, val_metric)


def train_test_wandb(MODEL, config, x_path, y_path, seed, weight_path, log_metric=True, log_grad=False, project=None, group=None, job_type=None, val_metric="loss"):
    """
    Args:
        MODEL:
        config:
        x_path: file path to data
        y_path: file path to label
        seed:
        weight_path:
        log: whether to log metrics to wandb
    """
    with wandb.init(config=config, project=project, group=group, job_type=job_type):
        config = wandb.config
        train_test(MODEL, config, x_path, y_path, seed, weight_path, log_metric, log_grad, val_metric)