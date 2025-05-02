import torch
import wandb
import argparse
import yaml
from train_eval import run, run_wandb
from models import Cnn, EcCnn_i, EcCnn, SigEcCnn


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)


parser = argparse.ArgumentParser()
parser.add_argument("--model_flag", help="Name of model to train")
parser.add_argument("--param", help="")
args = parser.parse_args()


model_dict = {
    "Cnn": Cnn,
    "EcCnn_i": EcCnn_i,
    "EcCnn": EcCnn,
    "SigEcCnn": SigEcCnn
    }


# load configuration file needed for training model
with open(f"ablation_configs/{args.model_flag}{args.param}.yaml", "r") as f:
    cfg = yaml.load(f, yaml.FullLoader)
cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    nsim = 15                   # number of simulations to run
    # cn_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]    # corruption and noise probabilities
    cn_prob_list = [0.0]    # corruption and noise probabilities

    wandb.login()
    project = "ablation"           # used as project name in wandb
    group = args.model_flag     # used for grouping experiments in wandb

    # loop over different noise probability
    for p in cn_prob_list:
        prob = str(int(p * 100)).zfill(2)
        job_type = prob                             # used for grouping experiments in wandb
        data_dir = f"./dataset/processed/{prob}/"   # base directory path to where data is loaded

        print("-"*30)
        print(f"Corruption & noise rate: {p}")
        print("-"*30)

        # weight_dir = f"./saved_weights/{args.model_flag}/{prob}/"    # directory path to save trained weights
        # os.makedirs(weight_dir, exist_ok=True)
        
        # loop over number of simulations
        for sim in range(1, nsim+1):
            print(f"\nSimulation: [{sim} / {nsim}]")
            print("-"*30)
            
            # weight_path = weight_dir + f"sim{sim}.pt"   # file path to save trained weights
            name = f"sim{sim}"  # used for specifying runs in wandb

            model = model_dict[args.model_flag](**cfg["model_params"]).to(cfg["device"])
            
            run_wandb(model, cfg, data_dir, project, group, job_type, name)
            # run(model, cfg, data_dir)