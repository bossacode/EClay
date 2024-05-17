import torch
import torch.nn as nn
from dtm import WeightedDTMLayer
from eclay import ECLay, GThetaEC
from pllay import PLLay, GThetaPL
from dect import EctLayer


# CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())
        # self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        #                                 nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(784, 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        x = self.conv_layer(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ECCNN(CNN):
    def __init__(self, in_channels=1, num_classes=10, hidden_features=[32]):
        super().__init__(in_channels, num_classes)
        self.gtheta_1 = GThetaEC(num_features=[32] + hidden_features)
        self.gtheta_2 = GThetaEC(num_features=[32] + hidden_features)
        self.fc = nn.Sequential(nn.Linear(784 + 2*hidden_features[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        # CNN
        x_1 = self.flatten(self.conv_layer(x))
        
        # EC Layer 1
        x_2 = self.gtheta_1(self.flatten(ecc_dtm005))

        # EC Layer 2
        x_3 = self.gtheta_2(self.flatten(ecc_dtm02))

        # FC Layer
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x


class ECCNN_Topo(CNN):
    def __init__(self, in_channels=1, num_classes=10,                                                                               # CNN params
                 as_vertices=False, sublevel=False, size=[28, 28], interval=[-7, 0], steps=32, hidden_features=[32], scale=0.1):    # EC params
        super().__init__(in_channels, num_classes)
        self.gtheta_1 = GThetaEC(num_features=[32] + hidden_features)
        self.gtheta_2 = GThetaEC(num_features=[32] + hidden_features)
        self.topo_layer_3 = ECLay(as_vertices, sublevel, size, interval, steps, in_channels, hidden_features, scale=scale)
        self.fc = nn.Sequential(nn.Linear(784 + 3*hidden_features[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        # CNN
        x_1 = self.conv_layer(x)
        
        # EC Layer 1
        x_2 = self.gtheta_1(self.flatten(ecc_dtm005))

        # EC Layer 2
        x_3 = self.gtheta_2(self.flatten(ecc_dtm02))

        # EC Layer 3
        x_4 = self.topo_layer_3(x_1)

        # FC Layer
        x = torch.concat((self.flatten(x_1), x_2, x_3, x_4), dim=-1)
        x = self.fc(x)
        return x


class PLCNN(CNN):
    def __init__(self, in_channels=1, num_classes=10, hidden_features=[32]):
        super().__init__(in_channels, num_classes)
        self.gtheta_1 = GThetaPL(num_features=[128] + hidden_features)
        self.gtheta_2 = GThetaPL(num_features=[192] + hidden_features)
        self.fc = nn.Sequential(nn.Linear(784 + 2*hidden_features[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        # CNN
        x_1 = self.flatten(self.conv_layer(x))
        
        # PL Layer 1
        x_2 = self.gtheta_1(self.flatten(pl_dtm005))

        # PL Layer 2
        x_3 = self.gtheta_2(self.flatten(pl_dtm02))

        # FC Layer
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x


class PLCNN_Topo(CNN):
    def __init__(self, in_channels=1, num_classes=10,                                                                               # CNN params
                 as_vertices=False, sublevel=False, interval=[-7, 0], steps=32, K_max=2, dimensions=[0, 1], hidden_features=[32]):  # EC params
        super().__init__(in_channels, num_classes)
        self.gtheta_1 = GThetaPL(num_features=[128] + hidden_features)
        self.gtheta_2 = GThetaPL(num_features=[192] + hidden_features)
        self.topo_layer_3 = PLLay(as_vertices, sublevel, interval, steps, K_max, dimensions, in_channels, hidden_features)
        self.fc = nn.Sequential(nn.Linear(784 + 3*hidden_features[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        # CNN
        x_1 = self.conv_layer(x)
        
        # EC Layer 1
        x_2 = self.gtheta_1(self.flatten(pl_dtm005))

        # EC Layer 2
        x_3 = self.gtheta_2(self.flatten(pl_dtm02))

        # EC Layer 3
        x_4 = self.topo_layer_3(x_1)

        # FC Layer
        x = torch.concat((self.flatten(x_1), x_2, x_3, x_4), dim=-1)
        x = self.fc(x)
        return x


# class ECCNN_TopoDTM(CNN):
#     def __init__(self, in_channels=1, num_classes=10,                                                               # CNN params
#                  as_vertices=False, sublevel=True, size=[28, 28], interval=[-7, 0, 32], hidden_features=[32]):     # EC params
#         super().__init__(in_channels, num_classes)
#         self.gtheta_1 = GThetaEC(num_features=[32] + hidden_features)
#         self.gtheta_2 = GThetaEC(num_features=[32] + hidden_features)
#         self.dtm = WeightedDTMLayer(m0=0.05)
#         self.topo_layer_3 = ECLay(as_vertices, sublevel, size, interval, in_channels=1, hidden_features=[interval[-1]] + hidden_features)
#         self.fc = nn.Sequential(nn.Linear(784 + 3*hidden_features[-1], 64),
#                                 nn.ReLU(),
#                                 nn.Linear(64, num_classes))

#     def forward(self, x):
#         x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
#         # CNN
#         x_1 = self.conv_layer(x)
        
#         # EC Layer 1
#         x_2 = self.gtheta_1(self.flatten(ecc_dtm005))

#         # EC Layer 2
#         x_3 = self.gtheta_2(self.flatten(ecc_dtm02))

#         # EC Layer 3
#         x_4 = self.topo_layer_3(self.dtm(x_1))

#         # FC Layer
#         x = torch.concat((self.flatten(x_1), x_2, x_3, x_4), dim=-1)
#         x = self.fc(x)
#         return x


class EctCnnModel(CNN):
    def __init__(self, in_channels=1, num_classes=32,
                 bump_steps=32, num_features=3, num_thetas=32, R=1.1, ect_type="faces", device="cpu", fixed=False):
        super().__init__(in_channels, num_classes)
        self.ectlayer = EctLayer(bump_steps, num_features, num_thetas, R, ect_type, device, fixed)
        self.fc = nn.Sequential(nn.Linear(1024, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, num_classes))

    def forward(self, batch):
        x = self.ectlayer(batch).unsqueeze(1)
        x = self.conv_layer(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x