import torch
import os
import wandb
import argparse
import yaml
from models import *
from train_test import train_test_wandb, train_test


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of model to train")
args = parser.parse_args()

models = {"CNN": CNN,
          "ECCNN": ECCNN,
          "ECCNN_Topo": ECCNN_Topo,
          "ECCNN_TopoDTM": ECCNN_TopoDTM,
          "ECResNet": ECResNet,
          "PLCNN": PLCNN,
          "PLResNet": PLResNet,
          "ResNet": ResNet}

# load configuration file needed for training model
with open(f"configs/{args.model}.yaml", "r") as f:
    config = yaml.load(f, yaml.FullLoader)
config["device"] = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    ntimes = 50  # number of simulations to run
    # noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    noise_prob_list = [0.0]
    n_train_list = [100, 300, 500, 700, 1000]    # list of number of training samples for each label
    # n_train_list = [100]

    wandb.login()
    for n_train in n_train_list:
        project = f"MNIST_data_" + args.model    # used as project name in wandb
        # project = f"MNIST_noise_" + args.model    # used as project name in wandb
        print("-"*30)
        print(f"Number of training data: {n_train}")
        print("-"*30)
        # loop over data with different noise probability
        for p in noise_prob_list:
            noise_prob = str(int(p * 100)).zfill(2)
            weight_dir = f"./saved_weights/data_{n_train}/{args.model}/{noise_prob}/"    # directory path to save trained weights
            os.makedirs(weight_dir, exist_ok=True)
            
            print("-"*30)
            print(f"Noise rate: {noise_prob}")
            print("-"*30)
            
            # loop over number of simulations
            for n_sim in range(1, ntimes+1):
                print(f"\nSimulation: [{n_sim} / {ntimes}]")
                print("-"*30)
                
                dir_path = f"./dataset/processed/data_{n_train}/noise_{noise_prob}/" # directory path to data
                weight_path = weight_dir + f"sim{n_sim}.pt"                     # file path to save trained weights
                # group = noise_prob                                              # used for grouping experiments in wandb
                group = str(n_train)
                job_type = f"sim{n_sim}"                                        # used for specifying runs in wandb        
                
                train_test_wandb(models[args.model], config, dir_path, weight_path, True, False, project, group, job_type)
                # train_test(models[args.model], config, dir_path, weight_path)