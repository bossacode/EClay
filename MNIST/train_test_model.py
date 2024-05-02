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
          "ECResNet": ECResNet,
          "PLCNN": PLCNN,
          "PLResNet": PLResNet,
          "ResNet": ResNet}

# load configuration file needed for training model
with open(f"configs/{args.model}.yaml", "r") as f:
    config = yaml.load(f, yaml.FullLoader)
config["device"] = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    ntimes = 100  # number of simulations to run
    noise_prob_list = [0.0]
    # noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    n_train_list = [10]
    # n_train_list = [10, 30, 50, 70, 100]    # list of number of training samples for each label

    wandb.login()
    for n_train in n_train_list:
        project = f"MNIST{n_train}_" + args.model    # used as project name in wandb
        print("-"*30)
        print(f"Number of data for each label: {n_train}")
        print("-"*30)
        # loop over data with different noise probability
        for p in noise_prob_list:
            noise_prob = str(int(p * 100)).zfill(2)
            weight_dir = f"saved_weights/data_{n_train}/{args.model}/{noise_prob}/"    # directory path to save trained weights
            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)
            
            print("-"*30)
            print(f"Noise rate: {noise_prob}")
            print("-"*30)
            
            # loop over number of simulations
            for n_sim in range(1, ntimes+1):
                print(f"\nSimulation: [{n_sim} / {ntimes}]")
                print("-"*30)
                
                dir_path = f"generated_data/data_{n_train}/noise_{noise_prob}/" # directory path to data
                weight_path = weight_dir + f"sim{n_sim}.pt"                     # file path to save trained weights
                group = noise_prob                                              # used for grouping experiments in wandb
                job_type = f"sim{n_sim}"                                        # used for specifying runs in wandb        
                
                train_test_wandb(models[args.model], config, dir_path, weight_path, True, False, project, group, job_type)
                # train_test(models[args.model], config, dir_path, weight_path)