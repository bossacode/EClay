import tensorflow as tf
import os
import wandb
import argparse
import yaml
from train_tf import set_dataloader, train_test, train_test_wandb
from models import PersCnnDTM, PlCnnDTM_i, PlCnnDTM


# for reproducibility (may degrade performance)
tf.keras.utils.set_random_seed(123)
tf.config.experimental.enable_op_determinism()


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of model to train")
args = parser.parse_args()


models = {
    "PersCnnDTM": PersCnnDTM,
    "PlCnnDTM_i": PlCnnDTM_i,
    "PlCnnDTM": PlCnnDTM
    }


# load configuration file needed for training model
with open(f"configs/KMNIST/{args.model}.yaml", "r") as f:
    cfg = yaml.load(f, yaml.FullLoader)


if __name__ == "__main__":
    nsim = 15                                       # number of simulations to run
    cn_prob_list = [0.05, 0.1, 0.15, 0.2, 0.25]     # corruption and noise probabilities

    wandb.login()

    # loop over different noise probability
    for p in cn_prob_list:
        project = "KMNIST_noise_shallow"     # used as project name in wandb

        print("-"*30)
        print(f"Corruption & noise rate: {p}")
        print("-"*30)

        prob = str(int(p * 100)).zfill(2)
        data_dir = "../KMNIST/dataset/processed/cn_prob/"                       # base directory path to where data is loaded
        weight_dir = f"./saved_weights/{args.model}/cn_prob/{prob}/"    # directory path to save trained weights
        os.makedirs(weight_dir, exist_ok=True)
        
        # loop over number of simulations
        for sim in range(1, nsim+1):
            print(f"\nSimulation: [{sim} / {nsim}]")
            print("-"*30)
            
            weight_path = weight_dir + f"sim{sim}.npz"   # file path to save trained weights
            group = args.model                          # used for grouping experiments in wandb
            job_type = prob                             # used for grouping experiments in wandb
            name = f"sim{sim}"                          # used for specifying runs in wandb
        
            train_dl, val_dl, test_dl = set_dataloader(data_dir + f"{prob}/train.pt", data_dir + f"{prob}/val.pt", data_dir + f"{prob}/test.pt", cfg["batch_size"])

            model = models[args.model](**cfg["model_params"])
            optim = tf.keras.optimizers.Adam(learning_rate=cfg["lr"], weight_decay=0.0001)
            train_test_wandb(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, True, False, project, group, job_type, name)
            # train_test(model, cfg, optim, train_dl, val_dl, test_dl, weight_path)