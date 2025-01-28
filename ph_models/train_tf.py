import sys
sys.path.append("../")
import tensorflow as tf
from keras import backend as K
import numpy as np
import torch
from sklearn.metrics import classification_report
from time import time
import wandb
from utils.train import EarlyStopping


def permute_dataset(dataset):
    *X, y = dataset
    dataset = tf.data.Dataset.from_tensor_slices((*[x.permute(0,2,3,1) if x.ndim == 4 else x for x in X], y))
    return dataset.shuffle(100, reshuffle_each_iteration=True, seed=123)


def set_dataloader(train_file_path, val_file_path, test_file_path, batch_size):
    train_dataset = torch.load(train_file_path, weights_only=True)
    train_dl = permute_dataset(train_dataset).batch(batch_size)
    val_dataset = torch.load(val_file_path, weights_only=True)
    val_dl = permute_dataset(val_dataset).batch(batch_size)
    test_dataset = torch.load(test_file_path, weights_only=True)
    test_dl = permute_dataset(test_dataset).batch(batch_size)
    return train_dl, val_dl, test_dl


class ReduceLROnPlateau:
    def __init__(self, factor, patience, threshold):
        self.factor = factor
        self.patiece = patience
        self.threshold = threshold
        self.count = 0
        self.best_loss = float("inf")

    def reduce_lr(self, val_loss, optim):
        diff = (self.best_loss - val_loss)
        if diff > self.threshold:
            self.count = 0
            self.best_loss = val_loss
        else:
            self.count += 1
            if self.count > self.patiece:
                K.set_value(optim.lr, optim.lr.numpy() * self.factor)
                print("-" * 100)
                print("Reducing learning rate to: ", optim.lr.numpy())
                print("-" * 100)
                self.count = 0


def train(model, dataloader, loss_fn, optimizer):
    """
    train for 1 epoch
    """
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for batch, (*X, y) in enumerate(dataloader, 1): # X is a list containing batch of original data and DTM transformed data  
        with tf.GradientTape() as tape:
            y_pred = model(X, training=True)
            loss = loss_fn(y_true=y, y_pred=y_pred)

        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        epoch_loss_avg.update_state(loss)
        epoch_accuracy.update_state(y, y_pred)

        if batch % 10 == 1:
            print(f"Training loss: {loss.numpy():>7f} [{batch*len(y):>3d}/?]")
    print(f"Train error:\n Accuracy: {epoch_accuracy.result() * 100:>0.1f}%, Avg loss: {epoch_loss_avg.result():>8f} \n")
    return epoch_loss_avg.result(), epoch_accuracy.result()


def test(model, dataloader, loss_fn):
    """
    """
    y_pred_list, y_true_list = [], []
    avg_loss, correct, data_size = 0, 0, 0
    for *X, y in dataloader:
        y_pred = model(X, training=False)
        loss = loss_fn(y_true=y, y_pred=y_pred)
        avg_loss += (loss.numpy() * len(y))
        correct += (y_pred.numpy().argmax(1) == y.numpy()).sum()
        data_size += len(y)
        y_pred_list.append(y_pred.numpy().argmax(1))
        y_true_list.append(y)
    avg_loss /= data_size
    accuracy = (correct / data_size) * 100
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    predicted = np.concatenate(y_pred_list)
    ground_truth = np.concatenate(y_true_list)
    report = classification_report(ground_truth, predicted, zero_division="warn")
    print(report)
    return avg_loss, accuracy


def train_val(model, cfg, optim, train_dl, val_dl, weight_path=None, log_metric=False, log_grad=False, val_metric="loss"):
    # set loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # set learning rate scheduler
    try:
        scheduler = ReduceLROnPlateau(cfg["factor"], cfg["sch_patience"], cfg["threshold"])
    except KeyError:
        scheduler = None
        print("No learning rate scheduler.")

    # set early stopping
    es = EarlyStopping(cfg["es_patience"], cfg["threshold"], val_metric=val_metric)
    
    # train
    if log_grad:
        wandb.watch(model, loss_fn, log="all", log_freq=5)  # log gradients and model parameters every 5 batches
    start = time()
    for n_epoch in range(1, cfg["epochs"]+1):
        print(f"\nEpoch: [{n_epoch} / {cfg['epochs']}]")
        print("-"*30)

        train_loss, train_acc = train(model, train_dl, loss_fn, optim)
        val_loss, val_acc = test(model, val_dl, loss_fn)

        if scheduler:
            scheduler.reduce_lr(val_loss, optim)

        # early stopping
        stop, improvement = es.stop_training(val_loss, val_acc, n_epoch)
        if log_metric:
            wandb.log({"train":{"loss":train_loss, "accuracy":train_acc},
                    "val":{"loss":val_loss, "accuracy":val_acc},
                    "best_val":{"loss":es.best_loss, "accuracy":es.best_acc}}, step=n_epoch)
        if stop or n_epoch == cfg["epochs"]:
            end = time()
            training_time = end - start
            print(f"\nTraining time: {training_time}\n")
            if log_metric:
                wandb.log({"training_time": training_time})
                wandb.log({"best_epoch": es.best_epoch})
            if weight_path is not None:
                np.savez(weight_path, *models_weights)
            break
        elif improvement and weight_path is not None:
            models_weights = model.get_weights()


def train_test(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, log_metric=False, log_grad=False, val_metric="loss"):
    train_val(model, cfg, optim, train_dl, val_dl, weight_path, log_metric, log_grad, val_metric)
    # test
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loaded = np.load(weight_path)
    loaded_weights = [loaded[key] for key in loaded.files]
    model.set_weights(loaded_weights)
    test_loss, test_acc = test(model, test_dl, loss_fn)
    if log_metric: wandb.log({"test":{"loss":test_loss, "accuracy":test_acc}})


def train_val_wandb(model, cfg, optim, train_dl, val_dl, weight_path=None, log_metric=True, log_grad=False, project=None, group=None, job_type=None, name=None, val_metric="loss"):
    with wandb.init(config=cfg, project=project, group=group, job_type=job_type, name=name):
        cfg = wandb.config
        train_val(model, cfg, optim, train_dl, val_dl, weight_path, log_metric, log_grad, val_metric)


def train_test_wandb(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, log_metric=True, log_grad=False, project=None, group=None, job_type=None, name=None, val_metric="loss"):
    with wandb.init(config=cfg, project=project, group=group, job_type=job_type, name=name):
        cfg = wandb.config
        train_test(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, log_metric, log_grad, val_metric)