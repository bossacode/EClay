import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import os
from collections import defaultdict
from kmnist_models import ResNet18, PRNet18, AdaptivePRNet18
from kmnist_train import KMNISTCustomDataset, eval

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ntimes = 10         # number of repetition for simulation of each model
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    # corrupt_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # noise_prob_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    corrupt_prob_list = [0.0, 0.1, 0.2, 0.3]
    noise_prob_list = [0.0, 0.1, 0.2, 0.3]

    len_cn = len(corrupt_prob_list)
    file_cn_list = [None] * len_cn
    for cn in range(len_cn):
        file_cn_list[cn] = str(int(corrupt_prob_list[cn] * 100)).zfill(2) + "_" + str(int(noise_prob_list[cn] * 100)).zfill(2)
    x_dir_list = ["./generated_data/x_" + file_cn_list[i] + ".pt" for i in range(len_cn)]
    y_dir = "./generated_data/y.pt"

    model_list = [ResNet18, PRNet18, AdaptivePRNet18]

    run_name = "73_RN18_vs_PRN18_vs_APRN18"
    
    # test
    # loop over data with different corruption/noise probability
    for cn in range(len_cn):
        print("-"*30)
        print(f"Corruption rate: {corrupt_prob_list[cn]}")
        print(f"Noise rate: {noise_prob_list[cn]}")
        print("-"*30)
        test_info = defaultdict(list)    # defaultdict to store test info of (accuracy, loss)

        # loop over number of simulations
        for n_sim in range(ntimes):
            print(f"\nSimulation: [{n_sim+1} / {ntimes}]")
            print("-"*30)
            test_dataset = KMNISTCustomDataset(x_dir_list[cn], y_dir, mode="test")
            test_dataloader = DataLoader(test_dataset, batch_size)
            
            # loop over different models
            for MODEL in model_list:
                print(f"Model: {MODEL.__name__}")
                print("-"*30)
                weight_file = f"./saved_weights/{run_name}/x_{file_cn_list[cn]}/{MODEL.__name__}/sim{n_sim+1}.pt"     # file path to trained model weights
                model = MODEL().to(device)
                model.load_state_dict(torch.load(weight_file, map_location=device))
                test_loss, test_acc = eval(model, test_dataloader, loss_fn, device)

                # save test information
                test_info[MODEL.__name__].append({"sim" + str(n_sim+1):(test_acc, test_loss)})

                # write to tensorboard
                writer = SummaryWriter(f"./runs/{run_name}/{file_cn_list[cn]}/{MODEL.__name__}")
                writer.add_scalar(f"loss/sim", test_loss, n_sim+1)
                writer.add_scalar(f"accuracy", test_acc, n_sim+1)
                writer.flush()
            print("\n"*2)
        
        # save test info as json file
        test_info_dir = f"./test_info/{run_name}/x_{file_cn_list[cn]}"
        if not os.path.exists(test_info_dir):
            os.makedirs(test_info_dir)
        for MODEL in model_list:
            with open(test_info_dir + "/" + f"{MODEL.__name__}_test_info.json", "w", encoding="utf-8") as f:
                json.dump(test_info[MODEL.__name__], f, indent="\t")



