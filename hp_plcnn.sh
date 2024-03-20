# !/bin/bash

cd /root/pllay/EClay/

pip install -r "requirements.txt"

api_key="cb66d1605ac3cd7c944bbe89df3ca8c9bc6fe41e"
wandb login $api_key

python hp_plcnn.py MNIST -hid 32
python hp_plcnn.py MNIST -hid 64 32