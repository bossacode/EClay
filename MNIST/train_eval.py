import sys
sys.path.append("../")
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from time import time
from copy import deepcopy
import wandb
from utils.train_utils import set_dl, EarlyStopping


def train(model, dataloader, loss_fn, optim, device):
    """
    train for 1 epoch
    """
    data_size = len(dataloader.dataset)
    ma_loss, correct = 0, 0
    model.train()
    for batch, (x, x_dtm, y) in enumerate(dataloader, 1):
        X, y = (x.to(device), x_dtm), y.to(device)
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
        for x, x_dtm, y in dataloader:
            X, y = (x.to(device), x_dtm), y.to(device)
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


def run(model, cfg, data_dir, val_metric="loss", use_wandb=False):
    train_dl, val_dl, test_dl = set_dl(data_dir, cfg["batch_size"]) # set dataloader
    loss_fn = nn.CrossEntropyLoss()                 # set loss function
    optim = Adam(model.parameters(), lr=cfg["lr"])  # set optimizer
    # set learning rate scheduler
    try:
        scheduler = ReduceLROnPlateau(optim, mode="min" if val_metric == "loss" else "max",
                                      factor=cfg["factor"], patience=cfg["sch_patience"], threshold=cfg["threshold"], verbose=True)
    except KeyError:
        scheduler = None
        print("No learning rate scheduler.")
    es = EarlyStopping(cfg["es_patience"], cfg["threshold"], val_metric=val_metric) # set early stopping

    # train
    # wandb.watch(model, loss_fn, log="all", log_freq=5)  # log gradients and model parameters every 5 batches
    start = time()
    for i_epoch in range(1, cfg["epochs"]+1):
        print(f"\nEpoch: [{i_epoch} / {cfg['epochs']}]")
        print("-"*30)

        train_loss, train_acc = train(model, train_dl, loss_fn, optim, cfg["device"])
        val_loss, val_acc = test(model, val_dl, loss_fn, cfg["device"])

        if scheduler:
            scheduler.step(val_loss if val_metric == "loss" else val_acc)

        # early stopping
        stop, improvement = es.stop_training(val_loss, val_acc, i_epoch)
        if use_wandb:
            wandb.log({"train":{"loss":train_loss, "accuracy":train_acc},
                    "val":{"loss":val_loss, "accuracy":val_acc},
                    "best_val":{"loss":es.best_loss, "accuracy":es.best_acc}}, step=i_epoch)
        if stop or i_epoch == cfg["epochs"]:
            end = time()
            training_time = end - start
            print(f"\nTraining time: {training_time}\n")
            # torch.save(best_model_state, weight_path)   # save model weights
            if use_wandb:
                wandb.log({"training_time": training_time})
                wandb.log({"best_epoch": es.best_epoch})
            break
        elif improvement:
            best_model_state = deepcopy(model.state_dict())

    # test
    model.load_state_dict(best_model_state)
    test_loss, test_acc = test(model, test_dl, loss_fn, cfg["device"])
    if use_wandb:
        wandb.log({"test":{"loss":test_loss, "accuracy":test_acc}})


def run_wandb(model, cfg, data_dir, project=None, group=None, job_type=None, name=None, val_metric="loss"):
    with wandb.init(config=cfg, project=project, group=group, job_type=job_type, name=name):
        cfg = wandb.config
        run(model, cfg, data_dir, val_metric, use_wandb=True)