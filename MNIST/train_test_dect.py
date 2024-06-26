import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from models import *
from train_test import EarlyStopping
import wandb


def train(model, dataloader, loss_fn, optimizer, device):
    """
    train for 1 epoch
    """
    data_size = len(dataloader.dataset)
    ma_loss, correct = 0, 0
    model.train()
    for batch, data in enumerate(dataloader, 1): # X is a list containing batch of original data and DTM transformed data  
        data = data.to(device)
        y_pred = model(data)
        loss = loss_fn(y_pred, data.y)
        ma_loss += (loss.item() * len(data.y))  # bc. loss_fn predicts avg loss
        correct += (y_pred.argmax(1) == data.y).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 1:
            print(f"Training loss: {loss.item():>7f} [{batch*len(data.y):>3d}/{data_size:>3d}]")
    ma_loss /= data_size                    # moving average of loss over 1 epoch
    ma_acc = (correct / data_size) * 100    # moving average of accuracy over 1 epoch
    print(f"Train error:\n Accuracy: {ma_acc:>0.1f}%, Avg loss: {ma_loss:>8f} \n")
    return ma_loss, ma_acc


def eval(model, dataloader, loss_fn, device):
    """
    """
    y_pred_list, y_true_list = [], []
    data_size = len(dataloader.dataset)
    avg_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            y_pred = model(data)
            loss = loss_fn(y_pred, data.y)
            avg_loss += (loss.item() * len(data.y))
            correct += (y_pred.argmax(1) == data.y).sum().item()
            y_pred_list.append(y_pred.argmax(1))
            y_true_list.append(data.y)
    avg_loss /= data_size
    accuracy = (correct / data_size) * 100
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    predicted = torch.concat(y_pred_list).to("cpu")
    ground_truth = torch.concat(y_true_list).to("cpu")
    report = classification_report(ground_truth, predicted, zero_division="warn")
    print(report)
    return avg_loss, accuracy


def train_val(MODEL, config, train_dl, val_dl, weight_path=None, log_metric=False, log_grad=False, val_metric="loss"):
    """_summary_

    Args:
        MODEL (_type_): _description_
        config (_type_): _description_
        dir_path (_type_): _description_
        weight_path (_type_, optional): _description_. Defaults to None.
        log_metric (bool, optional): _description_. Defaults to False.
        log_grad (bool, optional): _description_. Defaults to False.
        val_metric (str, optional): _description_. Defaults to "loss".
    """
    model = MODEL(**config["model_params"]).to(config["device"])
    loss_fn = nn.CrossEntropyLoss()
    optim  = Adam(model.parameters(), lr=config["lr"], weight_decay=0)
    scheduler = ReduceLROnPlateau(optim, mode="min" if val_metric == "loss" else "max",
                                  factor=config["factor"], patience=config["sch_patience"], threshold=config["threshold"], verbose=True)
    es = EarlyStopping(config["es_patience"], config["threshold"], val_metric=val_metric)

    if log_grad:
        wandb.watch(model, loss_fn, log="all", log_freq=5)  # log gradients and model parameters every 5 batches

    # train
    for n_epoch in range(1, config["epochs"]+1):
        print(f"\nEpoch: [{n_epoch} / {config['epochs']}]")
        print("-"*30)

        train_loss, train_acc = train(model, train_dl, loss_fn, optim, config["device"])
        val_loss, val_acc = eval(model, val_dl, loss_fn, config["device"])

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
            if log_metric:
                wandb.log({"best_epoch": es.best_epoch})
            break
        elif improvement and weight_path is not None:
            model_state_dict = model.state_dict()


def train_test(MODEL, config, train_dl, val_dl, test_dl, weight_path, log_metric=False, log_grad=False, val_metric="loss"):
    """_summary_

    Args:
        MODEL (_type_): _description_
        config (_type_): _description_
        dir_path (_type_): _description_
        weight_path (_type_): _description_
        log_metric (bool, optional): _description_. Defaults to False.
        log_grad (bool, optional): _description_. Defaults to False.
        val_metric (str, optional): _description_. Defaults to "loss".
    """
    train_val(MODEL, config, train_dl, val_dl, weight_path, log_metric, log_grad, val_metric)
    
    # test
    loss_fn = nn.CrossEntropyLoss()
    model = MODEL(**config["model_params"]).to(config["device"])
    model.load_state_dict(torch.load(weight_path, map_location=config["device"]))
    test_loss, test_acc = eval(model, test_dl, loss_fn, config["device"])
    if log_metric: wandb.log({"test":{"loss":test_loss, "accuracy":test_acc}})


def train_val_wandb(MODEL, config, train_dl, val_dl, weight_path=None, log_metric=True, log_grad=False, project=None, group=None, job_type=None, val_metric="loss"):
    """_summary_

    Args:
        MODEL (_type_): _description_
        config (_type_): _description_
        dir_path (_type_): _description_
        weight_path (_type_, optional): _description_. Defaults to None.
        log_metric (bool, optional): _description_. Defaults to True.
        log_grad (bool, optional): _description_. Defaults to False.
        project (_type_, optional): _description_. Defaults to None.
        group (_type_, optional): _description_. Defaults to None.
        job_type (_type_, optional): _description_. Defaults to None.
        val_metric (str, optional): _description_. Defaults to "loss".
    """
    with wandb.init(config=config, project=project, group=group, job_type=job_type):
        config = wandb.config
        train_val(MODEL, config, train_dl, val_dl, weight_path, log_metric, log_grad, val_metric)


def train_test_wandb(MODEL, config, train_dl, val_dl, test_dl, weight_path, log_metric=True, log_grad=False, project=None, group=None, job_type=None, val_metric="loss"):
    """_summary_

    Args:
        MODEL (_type_): _description_
        config (_type_): _description_
        dir_path (_type_): _description_
        weight_path (_type_): _description_
        log_metric (bool, optional): _description_. Defaults to True.
        log_grad (bool, optional): _description_. Defaults to False.
        project (_type_, optional): _description_. Defaults to None.
        group (_type_, optional): _description_. Defaults to None.
        job_type (_type_, optional): _description_. Defaults to None.
        val_metric (str, optional): _description_. Defaults to "loss".
    """
    with wandb.init(config=config, project=project, group=group, job_type=job_type):
        config = wandb.config
        train_test(MODEL, config, train_dl, val_dl, test_dl, weight_path, log_metric, log_grad, val_metric)