import torch
import wandb
import argparse
import os
from functools import partial
from models import EC_CNN_2
from train_test import train_val_wandb


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("data", help="directory of the dataset")
# parser.add_argument("-d", "--data", help="directory of the dataset")
# parser.add_argument("-t", "--T", nargs="+", type=int, help="")
# parser.add_argument("-hid", "--hidden", nargs="+", type=int, help="hidden feature dimensions of PLLay")
args = parser.parse_args()

project = "hp_ECCNN_" + args.data

sweep_config = {
    # "method": "random",
    "method": "grid",
    "metric": {"name": "best_val.loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"value": 16},
        "device": {"value": "cuda" if torch.cuda.is_available() else "cpu"},
        "epochs": {"value": 500},
        "factor":{"values": [0.1, 0.3, 0.5]},
        "lr_conv": {"values": [0.0001, 0.0005, 0.001, 0.005]},
        "lr_topo": {"values": [0.0001, 0.0005, 0.001, 0.005]},
        "lr_fc": {"values": [0.0001, 0.0005, 0.001, 0.005]},
        "model_params": {
            "parameters": dict(
                in_channels={"value": 1},
                num_classes={"value": 10},
                use_dtm={"value": True},
                # DTM parameters
                m0_1={"value": 0.05},
                m0_2={"value": 0.2},
                lims={"value": [[1,28], [1,28]]},
                size={"value": [28, 28]},
                r={"value": 2},
                # EC parameters
                start={"value": 0},
                end={"value": 7},
                T={"value": 32},
                num_channels={"value": 1},
                hidden_features={"value": [64, 32]},
                # EC parameters 2
                start_2={"value": 1},
                end_2={"value": 8}
                ),
        },
        # "es_patience": {"value": 35},
        "sch_patience": {"values": [10, 20]},
        "threshold": {"value": 0.001},
        "val_size": {"value": 0.8}
    },
    "description": f"Hyperparameter search of CNN on {args.data} using 100/400 train data"
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
    train_pipeline = partial(train_val_wandb, EC_CNN_2, None, x_path_list[0], y_path, seed, project=project)
    
    # wandb.agent(sweep_id, train_pipeline, project=project, count=150)
    wandb.agent(sweep_id, train_pipeline, project=project)