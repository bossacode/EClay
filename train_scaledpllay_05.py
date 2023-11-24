import torch
import os
import wandb
import argparse
from models_base import ScaledPllay_05
from train_test import train_test_pipeline


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("data", help="directory of the dataset")
args = parser.parse_args()

project = "ScaledPllay_05_" + args.data

# hyperparams, model params, metadata, etc.
config = {
    "batch_size": 32,
    "lr": 0.05,
    "weight_decay":0.0001,
    "factor": 0.1,          # factor to decay lr by when loss stagnates
    "threshold": 0.005,     # min value to be considered as improvement in loss
    "es_patience": 15,      # earlystopping patience
    "sch_patience": 5,     # lr scheduler patience
    "epochs": 500,
    "val_size": 0.3,
    "ntimes": 5,            # number of repetitions for simulation of each model
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_params": dict(
        in_channels=3,
        num_classes=7,
        out_features=50,
        m0=0.05,
        T=25,
        lims=[[1,56], [1,56]],
        size=[56, 56],
        r=2,
        K_max=2,
        dimensions=[0, 1],
        device="cuda" if torch.cuda.is_available() else "cpu"
        ),
    "data": args.data,
    "model": "BasePllay 0.05"
    }

corrupt_prob_list = [0.0, 0.1, 0.2]
noise_prob_list = [0.0, 0.1, 0.2]
# corrupt_prob_list = [0.0]
# noise_prob_list = [0.0]
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
    wandb.login()

    # train
    # loop over data with different corruption/noise probability
    for i_cn in range(len_cn):
        print("-"*30)
        print(f"Corruption/Noise rate: {file_cn_list[i_cn]}")
        print("-"*30)
        
        # loop over number of simulations
        for n_sim in range(1, config["ntimes"]+1):
            print(f"\nSimulation: [{n_sim} / {config['ntimes']}]")
            print("-"*30)
            
            model = ScaledPllay_05(**config["model_params"]).to(config["device"])
            
            group = file_cn_list[i_cn]                              # used for grouping experiments in wandb
            job_type = f"sim{n_sim}"                                # used for specifying runs in wandb
            weight_path = weight_dir_list[i_cn] + f"sim{n_sim}.pt"  # file path to save trained weights
            seed = torch.randint(0,1000, size=(1,)).item()          # used for different train/val split in each simulataion
            train_test_pipeline(model, config, project, group, job_type, x_path_list[i_cn], y_path, weight_path, seed)