import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from time import time
import wandb


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.x, self.x_dtm, self.y = torch.load(file_path, weights_only=True)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.x_dtm[idx], self.y[idx]


def set_dataloader(train_file_path, val_file_path, test_file_path, batch_size):
    train_ds = CustomDataset(train_file_path)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    val_ds = CustomDataset(val_file_path)
    val_dl = DataLoader(val_ds, batch_size)
    
    test_ds = CustomDataset(test_file_path)
    test_dl = DataLoader(test_ds, batch_size)
    return train_dl, val_dl, test_dl


class EarlyStopping:
    def __init__(self, patience, threshold, val_metric="loss"):
        """_summary_

        Args:
            patience (_type_): _description_
            threshold (_type_): _description_
            val_metric (str, optional): _description_. Defaults to "loss".
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


def train(model, dataloader, loss_fn, optim, device):
    """
    train for 1 epoch
    """
    data_size = len(dataloader.dataset)
    ma_loss, correct = 0, 0
    model.train()
    for batch, (*X, y) in enumerate(dataloader, 1): # X is a list containing batch of original data and DTM transformed data  
        X, y = [i.to(device) for i in X], y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        ma_loss += (loss.item() * len(y))  # bc. loss_fn predicts avg loss
        correct += (y_pred.argmax(1) == y).sum().item()

        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch % 10 == 1:
            print(f"Training loss: {loss.item():>7f} [{batch*len(y):>3d}/{data_size:>3d}]")
    ma_loss /= data_size                    # moving average of loss over 1 epoch
    ma_acc = (correct / data_size) * 100    # moving average of accuracy over 1 epoch
    print(f"Train error:\n Accuracy: {ma_acc:>0.1f}%, Avg loss: {ma_loss:>8f} \n")
    return ma_loss, ma_acc


def test(model, dataloader, loss_fn, device):
    """
    """
    y_pred_list, y_true_list = [], []
    data_size = len(dataloader.dataset)
    avg_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for *X, y in dataloader:
            X, y = [i.to(device) for i in X], y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            avg_loss += (loss.item() * len(y))
            correct += (y_pred.argmax(1) == y).sum().item()
            y_pred_list.append(y_pred.argmax(1))
            y_true_list.append(y)
    avg_loss /= data_size
    accuracy = (correct / data_size) * 100
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    predicted = torch.concat(y_pred_list).to("cpu")
    ground_truth = torch.concat(y_true_list).to("cpu")
    report = classification_report(ground_truth, predicted, zero_division="warn")
    print(report)
    return avg_loss, accuracy


def train_val(model, cfg, optim, train_dl, val_dl, weight_path=None, log_metric=False, log_grad=False, val_metric="loss"):
    # set loss function
    loss_fn = nn.CrossEntropyLoss()

    # set learning rate scheduler
    try:
        scheduler = ReduceLROnPlateau(optim, mode="min" if val_metric == "loss" else "max",
                                      factor=cfg["factor"], patience=cfg["sch_patience"], threshold=cfg["threshold"], verbose=True)
    except KeyError:
        scheduler = None
        print("No learning rate scheduler.")
        
    # set early stopping
    es = EarlyStopping(cfg["es_patience"], cfg["threshold"], val_metric=val_metric)

    # train
    if log_grad:
        wandb.watch(model, loss_fn, log="all", log_freq=5)  # log gradients and model parameters every 5 batches
    if log_metric:
        start = time()
    for n_epoch in range(1, cfg["epochs"]+1):
        print(f"\nEpoch: [{n_epoch} / {cfg['epochs']}]")
        print("-"*30)

        train_loss, train_acc = train(model, train_dl, loss_fn, optim, cfg["device"])
        val_loss, val_acc = test(model, val_dl, loss_fn, cfg["device"])

        if scheduler:
            scheduler.step(val_loss if val_metric == "loss" else val_acc)

        # early stopping
        stop, improvement = es.stop_training(val_loss, val_acc, n_epoch)
        if log_metric:
            wandb.log({"train":{"loss":train_loss, "accuracy":train_acc},
                    "val":{"loss":val_loss, "accuracy":val_acc},
                    "best_val":{"loss":es.best_loss, "accuracy":es.best_acc}}, step=n_epoch)
        if stop:
            if log_metric:
                end = time()
                wandb.log({"training_time": end - start})
                wandb.log({"best_epoch": es.best_epoch})
            if weight_path is not None:
                torch.save(model_state_dict, weight_path)   # save model weights
            break
        elif improvement and weight_path is not None:
            model_state_dict = model.state_dict()


def train_test(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, log_metric=False, log_grad=False, val_metric="loss"):
    train_val(model, cfg, optim, train_dl, val_dl, weight_path, log_metric, log_grad, val_metric)
    
    # test
    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(weight_path, map_location=cfg["device"], weights_only=True))
    test_loss, test_acc = test(model, test_dl, loss_fn, cfg["device"])
    if log_metric: wandb.log({"test":{"loss":test_loss, "accuracy":test_acc}})


def train_val_wandb(model, cfg, optim, train_dl, val_dl, weight_path=None, log_metric=True, log_grad=False, project=None, group=None, job_type=None, name=None, val_metric="loss"):
    with wandb.init(config=cfg, project=project, group=group, job_type=job_type, name=name):
        cfg = wandb.config
        train_val(model, cfg, optim, train_dl, val_dl, weight_path, log_metric, log_grad, val_metric)


def train_test_wandb(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, log_metric=True, log_grad=False, project=None, group=None, job_type=None, name=None, val_metric="loss"):
    with wandb.init(config=cfg, project=project, group=group, job_type=job_type, name=name):
        cfg = wandb.config
        train_test(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, log_metric, log_grad, val_metric)