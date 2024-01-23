import torch
import wandb
import argparse
import os
from functools import partial
from models_base import EC_Pllay
from train_test import train_pipeline_wandb


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("data", help="directory of the dataset")
args = parser.parse_args()

project = "hp_search_EC_Pllay_" + args.data

sweep_config = {
    "method": "random",
    "metric": {"name": "val.loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "lr": {"distribution": "inv_log_uniform_values", "min": 0.001, "max": 0.1},
        "weight_decay": {"values": [0, 0.0001]},
        "factor":{"distribution": "uniform", "min": 0.1, "max": 0.5},
        "threshold": {"value": 0.005},
        # "es_patience": {"value": 35},
        "sch_patience": {"values": [5, 10, 15, 20]},
        "epochs": {"value": 500},
        "val_size": {"value": 0.3},
        "device": {"value": "cuda" if torch.cuda.is_available() else "cpu"},
        "model_params": {
            "parameters": {
                "num_classes": {"value": 10},
                "use_dtm": {"value": False},
                # DTM parameters
                # m0={"distribution": "uniform", "min": 0.01, "max": 0.1},
                # lims={"value": [[1,28], [1,28]]},
                # size={"value": [28, 28]},
                # r={"value": 2},
                # PL parameters
                "T": {"values":[100, 150]},
                "num_channels": {"value": 3},
                "out_features": {"values":[50, 100, 150]},
                "p": {"values":[0, 0.1, 0.2, 0.3, 0.4, 0.5]}
            }
        }
    },
    "description": f"Hyperparameter search of EC_Pllay on {args.data} using 1000 train data"
}


corrupt_prob_list = [0.0]
noise_prob_list = [0.0]
len_cn = len(corrupt_prob_list)
file_cn_list, weight_dir_list = [], []
for i_cn in range(len_cn):
    file_cn_list.append(str(int(corrupt_prob_list[i_cn] * 100)).zfill(2) + "_" + str(int(noise_prob_list[i_cn] * 100)).zfill(2))
    weight_dir = f"{args.data}/saved_weights/{project}/{file_cn_list[i_cn]}"  # directory path to store trained model weights
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    weight_dir_list.append(weight_dir)
x_path_list = [f"{args.data}/generated_data/x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
y_path = f"{args.data}/generated_data/y.pt"


if __name__ == "__main__":
    seed = torch.randint(0,1000, size=(1,)).item()
    
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=project)

    # wandb.agent returns empty config if functions with arguments are given
    # https://github.com/wandb/wandb/issues/2724
    train_pipeline = partial(train_pipeline_wandb, EC_Pllay, None, x_path_list[0], y_path, seed, project=project)
    
    wandb.agent(sweep_id, train_pipeline, project=project, count=300)