import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import os
from collections import defaultdict
# from kmnist_models import ResNet18, PRNet18, AdaptivePRNet18
from pretrain import CustomDataset, eval, MODEL, run_name, len_cn, file_cn_list, x_dir_list, y_dir, ntimes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    record_test_info = False
    record_tensorboard = False
    
    # test
    # loop over data with different corruption/noise probability
    for cn in range(len_cn):
        print("-"*30)
        print(f"Corruption/Noise rate: {file_cn_list[cn]}")
        print("-"*30)
        test_info = defaultdict(list)    # defaultdict to store test info of (accuracy, loss)

        # loop over number of simulations
        for n_sim in range(ntimes):
            print(f"\nSimulation: [{n_sim+1} / {ntimes}]")
            print("-"*30)
            test_dataset = CustomDataset(x_dir_list[cn], y_dir, mode="test")
            test_dataloader = DataLoader(test_dataset, batch_size)

            print(f"Model: {MODEL.__name__}")
            print("-"*30)
            weight_file = f"./saved_weights/{run_name}/{file_cn_list[cn]}/{MODEL.__name__}/sim{n_sim+1}.pt"     # file path to trained model weights
            model = MODEL().to(device)
            model.load_state_dict(torch.load(weight_file, map_location=device))
            test_loss, test_acc = eval(model, test_dataloader, loss_fn, device)

            if record_test_info:
                # save test information
                test_info[MODEL.__name__].append({"sim" + str(n_sim+1):(test_acc, test_loss)})

            if record_tensorboard:
            # write to tensorboard
                writer = SummaryWriter(f"./runs/{run_name}/test/{file_cn_list[cn]}/{MODEL.__name__}")
                writer.add_scalar(f"loss/sim", test_loss, n_sim+1)
                writer.add_scalar(f"accuracy", test_acc, n_sim+1)
                writer.flush()
            print("\n"*2)
        
        # save test info as json file
        if record_test_info:
            test_info_dir = f"./test_info/{run_name}/{file_cn_list[cn]}"
            if not os.path.exists(test_info_dir):
                os.makedirs(test_info_dir)
            with open(test_info_dir + "/" + f"{MODEL.__name__}_test_info.json", "w", encoding="utf-8") as f:
                json.dump(test_info[MODEL.__name__], f, indent="\t")



