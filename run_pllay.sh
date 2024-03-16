# !/bin/bash

cd /root/pllay/Functional_Pllay/

pip install -r "requirements.txt"

api_key="cb66d1605ac3cd7c944bbe89df3ca8c9bc6fe41e"
wandb login $api_key

python hp_pllay.py -d MNIST -t 128 64 32 -hid 32
python hp_pllay.py -d MNIST -t 128 64 32 -hid 64
python hp_pllay.py -d MNIST -t 128 64 32 -hid 128