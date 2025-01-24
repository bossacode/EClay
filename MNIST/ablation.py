import sys
sys.path.append("../")
import torch
from torch.optim import Adam
import os
import wandb
import argparse
import yaml
from utils.train import set_dataloader, train_test, train_test_wandb
from models import Cnn, EcCnn_i, EcCnn


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of model to train")
parser.add_argument("param", help="")
args = parser.parse_args()


models = {
    "Cnn": Cnn,
    "EcCnn_i": EcCnn_i,
    "EcCnn": EcCnn
    }


# load configuration file needed for training model
with open(f"ablation_configs/{args.model}{args.param}.yaml", "r") as f:
    cfg = yaml.load(f, yaml.FullLoader)
cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    nsim = 15                   # number of simulations to run
    train_size_list = [100, 1000]    # training sample sizes

    wandb.login()

    # loop over different training size
    for train_size in train_size_list:
        project = "test"      # used as project name in wandb
        
        print("-"*30)
        print(f"Number of training data: {train_size}")
        print("-"*30)

        data_dir = "./dataset/processed/train_size/"                            # base directory path to where data is loaded
        weight_dir = f"./saved_weights/{args.model}/train_size/{train_size}/"   # directory path to save trained weights
        os.makedirs(weight_dir, exist_ok=True)
        
        # loop over number of simulations
        for sim in range(1, nsim+1):
            print(f"\nSimulation: [{sim} / {nsim}]")
            print("-"*30)
            
            weight_path = weight_dir + f"sim{sim}.pt"   # file path to save trained weights
            group = args.model                          # used for grouping experiments in wandb
            job_type = str(train_size)                  # used for grouping experiments in wandb
            name = f"sim{sim}"                          # used for specifying runs in wandb

            train_dl, val_dl, test_dl = set_dataloader(data_dir + f"{train_size}/train.pt", data_dir + f"{train_size}/val.pt", data_dir + "test.pt", cfg["batch_size"])
            model = models[args.model](**cfg["model_params"]).to(cfg["device"])
            optim = Adam(model.parameters(), lr=cfg["lr"])
            train_test_wandb(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, True, False, project, group, job_type, name)
            # train_test(model, cfg, optim, train_dl, val_dl, test_dl, weight_path)