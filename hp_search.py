import torch
import os
import wandb
import argparse
import yaml
from functools import partial
from models import *
from train_test import train_val_wandb


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of model to train")
parser.add_argument("data", help="Name of data to use")
args = parser.parse_args()

models = {"CNN": CNN,
          "ECCNN": ECCNN,
          "ECNet": ECNet,
          "ECNet2": ECNet2,
          "ECResNet": ECResNet,
          "PLCNN": PLCNN,
          "PLNet": PLNet,
          "PLNet2": PLNet2,
          "PLResNet": PLResNet,
          "ResNet": ResNet}

with open(f"{args.data}/sweep_configs/{args.model}.yaml", "r") as f:
    sweep_config = yaml.load(f, yaml.FullLoader)
sweep_config["parameters"]["device"] = {"value": "cuda" if torch.cuda.is_available() else "cpu"}


if __name__ == "__main__":
    seed = torch.randint(0,1000, size=(1,)).item()
    
    wandb.login()
    project = "hp_" + args.model + "_" + args.data  # used as project name in wandb
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(sweep_config)

    dir_path = f"{args.data}/generated_data/noise_00/"
    # wandb.agent returns empty config if functions with arguments are given
    # https://github.com/wandb/wandb/issues/2724
    train_pipeline = partial(train_val_wandb, models[args.model], None, dir_path, project=project)
    
    # wandb.agent(sweep_id, train_pipeline, project=project, count=300)
    wandb.agent(sweep_id, train_pipeline, project=project)