# !/bin/bash

cd /root/pllay/Functional_Pllay/

pip install -r "requirements.txt"

api_key="cb66d1605ac3cd7c944bbe89df3ca8c9bc6fe41e"
wandb login $api_key

python hp_eclay.py -d MNIST -t 128 64 -hid 32
python hp_eclay.py -d MNIST -t 256 128 -hid 64
python hp_eclay.py -d MNIST -t 512 256 -hid 128