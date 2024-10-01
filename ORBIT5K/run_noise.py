import sys
sys.path.append("../")
import torch
from torch.optim import Adam
import os
import wandb
import argparse
import yaml
from utils.train import set_dataloader, train_test, train_test_wandb
from models import Cnn, EcCnn_i, EcCnn, EcCnnDTM_i, EcCnnDTM


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of model to train")
args = parser.parse_args()


models = {
    "Cnn": Cnn,
    "EcCnn_i": EcCnn_i,
    "EcCnn": EcCnn,
    "EcCnnDTM_i": EcCnnDTM_i,
    "EcCnnDTM": EcCnnDTM
    }


# load configuration file needed for training model
with open(f"configs/{args.model}.yaml", "r") as f:
    cfg = yaml.load(f, yaml.FullLoader)
cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"


def set_optimizer(model, cfg):
    param_list = []
    for name, layer in model.named_children():
        if "ecc" in name:   # set lr for all ECLayr
            print("name: ",name, "lr: ", cfg["lr_topo"])
            param_list.append({"params": layer.parameters(), "lr": cfg["lr_topo"]})
        else:
            print("name: ",name, "lr: ", cfg["lr"])
            param_list.append({"params": layer.parameters(), "lr": cfg["lr"]})
    optim = Adam(param_list, lr=cfg["lr"], weight_decay=0.0001)
    return optim


if __name__ == "__main__":
    nsim = 15                                   # number of simulations to run
    cn_prob_list = [0]    # corruption and noise probabilities
    cn_prob_list = [0.05, 0.1, 0.15, 0.2]    # corruption and noise probabilities

    wandb.login()

    # loop over different noise probability
    for p in cn_prob_list:
        project = "ORBIT5K_noise_shallow"     # used as project name in wandb

        print("-"*30)
        print(f"Corruption & noise rate: {p}")
        print("-"*30)

        prob = str(int(p * 100)).zfill(2)
        data_dir = f"./dataset/processed/noise_{prob}/"                 # base directory path to where data is loaded
        weight_dir = f"./saved_weights/{args.model}/noise_{prob}/"    # directory path to save trained weights
        os.makedirs(weight_dir, exist_ok=True)
        
        # loop over number of simulations
        for sim in range(1, nsim+1):
            print(f"\nSimulation: [{sim} / {nsim}]")
            print("-"*30)
            
            weight_path = weight_dir + f"sim{sim}.pt"   # file path to save trained weights
            group = args.model                          # used for grouping experiments in wandb
            job_type = prob                             # used for grouping experiments in wandb
            name = f"sim{sim}"                          # used for specifying runs in wandb
        
            train_dl, val_dl, test_dl = set_dataloader(data_dir + "train.pt", data_dir + "val.pt", data_dir + "test.pt", cfg["batch_size"])

            model = models[args.model](**cfg["model_params"]).to(cfg["device"])
            # optim = Adam(model.parameters(), lr=cfg["lr"])
            optim = set_optimizer(model, cfg)
            print(optim)
            train_test_wandb(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, True, False, project, group, job_type, name)
            # train_test(model, cfg, optim, train_dl, val_dl, test_dl, weight_path)