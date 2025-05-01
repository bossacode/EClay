import tensorflow as tf
import wandb
import argparse
import yaml
from train_tf import run, run_wandb
from ph_models.mnist_models import PersCnn, PlCnn_i, PlCnn


# for reproducibility (may degrade performance)
tf.keras.utils.set_random_seed(123)
tf.config.experimental.enable_op_determinism()


parser = argparse.ArgumentParser()
parser.add_argument("--model_flag", help="Name of model to train")
args = parser.parse_args()


models = {
    "PersCnn": PersCnn,
    "PlCnn_i": PlCnn_i,
    "PlCnn": PlCnn
    }


# load configuration file needed for training model
with open(f"configs/MNIST/{args.model_flag}.yaml", "r") as f:
    cfg = yaml.load(f, yaml.FullLoader)


if __name__ == "__main__":
    nsim = 20                                           # number of simulations to run
    cn_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]    # corruption and noise probabilities

    wandb.login()
    project = "MNIST"           # used as project name in wandb
    group = args.model_flag     # used for grouping experiments in wandb

    # loop over different noise probability
    for p in cn_prob_list:
        prob = str(int(p * 100)).zfill(2)
        job_type = prob                                     # used for grouping experiments in wandb
        data_dir = f"../MNIST/dataset/processed/{prob}/"    # base directory path to where data is loaded

        print("-"*30)
        print(f"Corruption & noise rate: {p}")
        print("-"*30)
        
        # weight_dir = f"./saved_weights/MNIST/{args.model_flag}/{prob}/"  # directory path to save trained weights
        # os.makedirs(weight_dir, exist_ok=True)
        
        # loop over number of simulations
        for sim in range(1, nsim+1):
            print(f"\nSimulation: [{sim} / {nsim}]")
            print("-"*30)
            
            # weight_path = weight_dir + f"sim{sim}.npz"   # file path to save trained weights
            name = f"sim{sim}"                          # used for specifying runs in wandb
            
            model = models[args.model_flag](**cfg["model_params"])
            
            run_wandb(model, cfg, data_dir, project, group, job_type, name)
            # train_test(model, cfg, data_dir)