import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, file_path):
        *self.X, self.y = torch.load(file_path, weights_only=True)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return *[x[idx] for x in self.X], self.y[idx]


def set_dl(data_dir, batch_size):
    train_ds = CustomDataset(data_dir + "train.pt")
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    val_ds = CustomDataset(data_dir + "val.pt")
    val_dl = DataLoader(val_ds, batch_size)
    
    test_ds = CustomDataset(data_dir + "test.pt")
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