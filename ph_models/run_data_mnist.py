import tensorflow as tf
import os
import wandb
import argparse
import yaml
from train_tf import set_dataloader, train_test, train_test_wandb
from ph_models.mnist_models import PersCnn, PlCnn_i, PlCnn


# for reproducibility (may degrade performance)
tf.keras.utils.set_random_seed(123)
tf.config.experimental.enable_op_determinism()


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of model to train")
args = parser.parse_args()


models = {
    "PersCnn": PersCnn,
    "PlCnn_i": PlCnn_i,
    "PlCnn": PlCnn
    }


# load configuration file needed for training model
with open(f"configs/MNIST/{args.model}.yaml", "r") as f:
    cfg = yaml.load(f, yaml.FullLoader)


if __name__ == "__main__":
    nsim = 20                                       # number of simulations to run
    train_size_list = [100, 300, 500, 700, 1000]    # training sample sizes

    wandb.login()

    # loop over different training size
    for train_size in train_size_list:
        project = "MNIST_data"  # used as project name in wandb
        
        print("-"*30)
        print(f"Number of training data: {train_size}")
        print("-"*30)

        data_dir = "../MNIST/dataset/processed/train_size/"                     # base directory path to where data is loaded
        weight_dir = f"./saved_weights/MNIST/{args.model}/train_size/{train_size}/"   # directory path to save trained weights
        os.makedirs(weight_dir, exist_ok=True)
        
        # loop over number of simulations
        for sim in range(1, nsim+1):
            print(f"\nSimulation: [{sim} / {nsim}]")
            print("-"*30)
            
            weight_path = weight_dir + f"sim{sim}.npz"   # file path to save trained weights
            group = args.model                          # used for grouping experiments in wandb
            job_type = str(train_size)                  # used for grouping experiments in wandb
            name = f"sim{sim}"                          # used for specifying runs in wandb

            train_dl, val_dl, test_dl = set_dataloader(data_dir + f"{train_size}/train.pt", data_dir + f"{train_size}/val.pt", data_dir + "test.pt", cfg["batch_size"])
            model = models[args.model](**cfg["model_params"])
            optim = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])
            train_test_wandb(model, cfg, optim, train_dl, val_dl, test_dl, weight_path, True, False, project, group, job_type, name)
            # train_test(model, cfg, optim, train_dl, val_dl, test_dl, weight_path)