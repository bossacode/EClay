import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
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


def fc_weight_norms(model):
    """
    get L1 weight norms for all fully connected layers
    """
    weight_norm_dict = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):    # gtheta layer, fc layer
            weight_norm = m.weight.detach().abs().sum(dim=0, keepdim=True).T        # L1 norm, shape: [in_dim, 1]
            weight_norm_dict[name] = weight_norm
    return weight_norm_dict


def fc_grad_norms(model):
    """
    get L1 gradient norms for all fully connected layers
    """
    grad_norm_dict = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):    # gtheta layer, fc layer
            grad_norm = m.weight.grad.detach().abs().sum(dim=0, keepdim=True).T     # L1 norm, shape: [in_dim, 1]
            grad_norm_dict[name] = grad_norm
    return grad_norm_dict


def train(model, dataloader, loss_fn, optimizer, device):
    """
    train for 1 epoch
    """
    data_size = len(dataloader.dataset)
    ma_loss, correct = 0, 0
    ma_grad_norm_dict = defaultdict(lambda: 0)
    model.train()
    for batch, (X, y) in enumerate(dataloader, 1):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        ma_loss += (loss.item() * len(y))  # bc. loss_fn predicts avg loss
        correct += (y_pred.argmax(1) == y).sum().item()

        loss.backward()
        optimizer.step()

        # grad norm
        grad_norm_dict =  fc_grad_norms(model)
        for name, grad_norm in grad_norm_dict.items():
            ma_grad_norm_dict[name] += (grad_norm * len(y))

        optimizer.zero_grad()

        if batch % 10 == 0:
            print(f"Training loss: {loss.item():>7f} [{batch*len(y):>3d}/{data_size:>3d}]")
    ma_loss /= data_size                        # moving average of loss over 1 epoch
    ma_acc = (correct / data_size) * 100        # moving average of accuracy over 1 epoch
    print(f"Train error:\n Accuracy: {ma_acc:>0.1f}%, Avg loss: {ma_loss:>8f} \n")

    for name, grad_norm in ma_grad_norm_dict.items():
        ma_grad_norm_dict[name] /= data_size    # moving average of gradient norms over 1 epoch
    return ma_loss, ma_acc, ma_grad_norm_dict


def eval(model, dataloader, loss_fn, device, mode):
    """
    mode: one of "val" or "test"
    """
    if mode == "test":
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
            if mode == "test":
                y_pred_list.append(y_pred.argmax(1))
    loss /= data_size
    accuracy = (correct / data_size) * 100
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")

    if mode == "test":
        predicted = torch.concat(y_pred_list).detach().to("cpu")
        ground_truth = dataloader.dataset.y_test
        report = classification_report(ground_truth, predicted, zero_division=0)
        print(report, "\n")
    return loss, accuracy


def train_test_pipeline(model, config, project, group, job_type, x_path, y_path, weight_path, seed):
    with wandb.init(project=project, group=group, job_type=job_type, config=config):
        config = wandb.config

        train_dataset = CustomDataset(x_path, y_path, mode="train", seed=seed, val_size=config.val_size)
        val_dataset = CustomDataset(x_path, y_path, mode="val", seed=seed, val_size=config.val_size)
        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, config.batch_size)
        
        loss_fn = nn.CrossEntropyLoss()
        optim = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = ReduceLROnPlateau(optim, factor=config.factor, patience=config.sch_patience, threshold=config.threshold, verbose=True)
        
        wandb.watch(model, loss_fn, log="all", log_freq=5)     # log gradients and model parameters every 5 batches
        
        best_loss, best_acc, best_epoch = float("inf"), None, None
        early_stop_counter = 0
        
        grad_norm_collection = defaultdict(list)
        weight_norm_collection = defaultdict(list)

        # train
        for n_epoch in range(1, config.epochs+1):
            print(f"\nEpoch: [{n_epoch} / {config.epochs}]")
            print("-"*30)

            train_loss, train_acc, ma_grad_norm_dict = train(model, train_dataloader, loss_fn, optim, config.device)
            val_loss, val_acc = eval(model, val_dataloader, loss_fn, config.device, mode="val")

            scheduler.step(val_loss)

            wandb.log({"train":{"loss":train_loss, "accuracy":train_acc},
                       "val":{"loss":val_loss, "accuracy":val_acc}}, step=n_epoch)
            
            # collect running average of gradients for every epoch
            for name, grad_norm in ma_grad_norm_dict.items():
                grad_norm_collection[name].append(grad_norm)
            
            # collect weights after every epoch
            weight_norm_dict = fc_weight_norms(model)
            for name, weight_norm in weight_norm_dict.items():
                weight_norm_collection[name].append(weight_norm)

            # early stopping (if loss improvement is below threshold, it's not considered as improvement)
            if val_loss < 2.5:  # to avoid early stopping in early stages of training
                if (best_loss - val_loss) > config.threshold:
                    early_stop_counter = 0
                    best_loss, best_acc, best_epoch = val_loss, val_acc, n_epoch
                    torch.save(model.state_dict(), weight_path)   # save model weights
                else:
                    early_stop_counter += 1
                    if early_stop_counter > config.es_patience:    # stop training if loss doesn't improve for es_patience + 1 epochs
                        print("-"*30)
                        print(f"Epochs: [{best_epoch} / {config.epochs}]")
                        print(f"Best Validation Accuracy: {(best_acc):>0.1f}%")
                        print(f"Best Validation Loss: {best_loss:>8f}")
                        print("-"*30)
                        break
        
        # log gradient norm images
        for name, grad_norm_list in grad_norm_collection.items():
            fig = plt.figure()
            plt.pcolor(torch.hstack(grad_norm_list))
            plt.colorbar()
            plt.title(name)
            plt.xlabel("epochs")
            plt.ylabel("nodes")
            wandb.log({"gradient chart": wandb.Image(fig)})

        # log weight norm images
        for name, weight_norm_list in weight_norm_collection.items():
            fig = plt.figure()
            plt.pcolor(torch.hstack(weight_norm_list))
            plt.colorbar()
            plt.title(name)
            plt.xlabel("epochs")
            plt.ylabel("nodes")
            wandb.log({"weight chart": wandb.Image(fig)})

        test_dataset = CustomDataset(x_path, y_path, mode="test")
        test_dataloader = DataLoader(test_dataset, config.batch_size)
        # test
        model.load_state_dict(torch.load(weight_path, map_location=config.device))
        test_loss, test_acc = eval(model, test_dataloader, loss_fn, config.device, mode="test")
        wandb.log({"test":{"loss":test_loss, "accuracy":test_acc}})