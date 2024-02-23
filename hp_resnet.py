import torch
import wandb
import argparse
import os
from functools import partial
from models import ResNet
from train_test import train_val_wandb

# python hp_resnet.py -d MNIST -b 2 2 2 -f 16 32 64

# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

parser = argparse.ArgumentParser()
# parser.add_argument("data", help="directory of the dataset")
parser.add_argument("-d", "--data", help="directory of the dataset")
parser.add_argument("-b", "--block_cfg", nargs="+", type=int, help="block configuration of ResNet")
parser.add_argument("-f", "--filter_cfg", nargs="+", type=int, help="filter configuration of ResNet")
args = parser.parse_args()

project = "hp_ResNet"

sweep_config = {
    "method": "random",
    "metric": {"name": "best_val.loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"value": 16},
        "device": {"value": "cuda" if torch.cuda.is_available() else "cpu"},
        "epochs": {"value": 200},
        "factor":{"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "lr": {"distribution": "inv_log_uniform_values", "min": 0.001, "max": 0.05},
        "model_params": {
            "parameters": dict(
                in_channels={"value": 1},
                # block=ResidualBlock,
                block_cfg={"value": args.block_cfg},
                filter_cfg={"value": args.filter_cfg},
                num_classes={"value": 10},
                p={"values": [0.1, 0.2, 0.3, 0.4, 0.5]}
            )
        },
        # "es_patience": {"value": 35},
        "sch_patience": {"values": [10, 15]},
        "threshold": {"value": 0.001},
        "val_size": {"value": 0.8}
    },
    "description": f"Hyperparam search of ResNet on {args.data} using 100/400 train/val data"
}


corrupt_prob_list = [0.0]
noise_prob_list = [0.0]
len_cn = len(corrupt_prob_list)
file_cn_list, weight_dir_list = [], []
for i_cn in range(len_cn):
    file_cn_list.append(str(int(corrupt_prob_list[i_cn] * 100)).zfill(2) + "_" + str(int(noise_prob_list[i_cn] * 100)).zfill(2))
    weight_dir = f"{args.data}/saved_weights/{project}/{file_cn_list[i_cn]}/"  # directory path to store trained model weights
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
    train_pipeline = partial(train_val_wandb, ResNet, None, x_path_list[0], y_path, seed, project=project)
    
    wandb.agent(sweep_id, train_pipeline, project=project, count=200)