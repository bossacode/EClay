import torch
import os
import wandb
import argparse
from models import ResidualBlock, EClayResNet
from train_test import train_test, train_test_wandb


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("data", help="directory of the dataset")
args = parser.parse_args()

project = "ECResNet_" + args.data 

# hyperparams, model params, metadata, etc.
config = {
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 200,
    "factor": 0.4,          # factor to decay lr by when loss stagnates
    "lr_res": 0.005,
    "lr_topo": 0.001,
    "lr_fc": 0.005,
    "model_params": dict(
        in_channels=1,
        # block=ResidualBlock,
        block_cfg=[2,2],
        filter_cfg=[64,128],
        num_classes=10,
        # loading pretrained ResNet
        # load_res=True,
        # res_path="./MNIST/saved_weights/ResNet_MNIST/00_00/sim1.pt",
        # freeze_res=True,
        # DTM parameters
        use_dtm=True,
        m0_1=0.05,
        m0_2=0.2,
        lims=[[1,28], [1,28]],
        size=[28, 28],
        r=2,
        # EC parameters
        start=0,
        end=7,
        T=32,
        num_channels=1,
        hidden_features=[64, 32],
        # EC parameters 2
        start_2=1,
        end_2=8,
        # loading pretrained ECLay
        # load_ec=True,
        # ec_path="./MNIST/saved_weights/EClay_MNIST/00_00/sim1.pt",
        # freeze_ec=True
        ),
    "ntimes": 1,            # number of repetitions for simulation of each model
    # "es_patience": 25,      # earlystopping patience
    "sch_patience": 10,     # lr scheduler patience
    "threshold": 0.001,     # min value to be considered as improvement in loss
    "val_size": 0.8
    }

# corrupt_prob_list = [0.0, 0.1, 0.2]
# noise_prob_list = [0.0, 0.1, 0.2]
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
            
            group = file_cn_list[i_cn]                              # used for grouping experiments in wandb
            job_type = f"sim{n_sim}"                                # used for specifying runs in wandb
            weight_path = weight_dir_list[i_cn] + f"sim{n_sim}.pt"  # file path to save trained weights
            seed = torch.randint(0,1000, size=(1,)).item()          # used for different train/val split in each simulataion
            # train_test_wandb(EClayResNet, config, x_path_list[i_cn], y_path, seed, weight_path, True, False, project, group, job_type)
            train_test(EClayResNet, config, x_path_list[i_cn], y_path, seed, weight_path)
