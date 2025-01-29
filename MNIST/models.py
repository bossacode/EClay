import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from eclayr.cubical.cubeclayr import CubEclayr, SigCubEclayr

topo_out_units = 64

# Cnn
class Cnn(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
            )
        self.fc = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x, x_dtm = x
        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Cnn + ECLayr
class EcCnn_i(Cnn):
    def __init__(self, in_channels=1, num_classes=10, *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.eclayr = CubEclayr(postprocess=nn.Linear(kwargs["steps"], topo_out_units), *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(784 + topo_out_units, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )

    def forward(self, x):
        x, x_dtm = x
        
        # ECLayr
        ecc = self.eclayr(x_dtm)
        ecc = F.relu(ecc)
        
        # CNN
        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, ecc), dim=-1)
        x = self.fc(x)
        return x


# Cnn + ECLayr + ECLayr after conv
class EcCnn(Cnn):
    def __init__(self, in_channels=1, num_classes=10, *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.eclayr_1 = CubEclayr(interval=kwargs["interval_1"], steps=kwargs["steps_1"], sublevel=kwargs["sublevel_1"], postprocess=nn.Linear(kwargs["steps_1"], topo_out_units), *args, **kwargs)
        self.eclayr_2 = CubEclayr(interval=kwargs["interval_2"], steps=kwargs["steps_2"], sublevel=kwargs["sublevel_2"], postprocess=nn.Linear(kwargs["steps_2"], topo_out_units), *args, **kwargs)
        self.fc = nn.Sequential(
                    nn.Linear(784 + 2*topo_out_units, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                    )

    def forward(self, x):
        x, x_dtm = x

        # first ECLayr
        ecc_1 = self.eclayr_1(x_dtm)
        ecc_1 = F.relu(ecc_1)

        # CNN
        x = self.conv(x)

        # second ECLayr after conv layer
        min_vals = x.amin(dim=(2, 3), keepdim=True)     # shape: [B, C, 1, 1]
        max_vals = x.amax(dim=(2, 3), keepdim=True)     # shape: [B, C, 1, 1]
        x_2 = (x - min_vals) / (max_vals - min_vals)    # normalize between 0 and 1 for each data and channel
        ecc_2 = self.eclayr_2(x_2)
        ecc_2 = F.relu(ecc_2)

        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, ecc_1, ecc_2), dim=-1)
        x = self.fc(x)
        return x


# Cnn + sigmoid ECLayr + sigmoid ECLayr after conv
class SigEcCnn(Cnn):
    def __init__(self, in_channels=1, num_classes=10, *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.sig_eclayr_1 = SigCubEclayr(interval=kwargs["interval_1"], steps=kwargs["steps_1"], sublevel=kwargs["sublevel_1"], lam=kwargs["lam_1"], postprocess=nn.Linear(kwargs["steps_1"], topo_out_units), *args, **kwargs)
        self.sig_eclayr_2 = SigCubEclayr(interval=kwargs["interval_2"], steps=kwargs["steps_2"], sublevel=kwargs["sublevel_2"], lam=kwargs["lam_2"], postprocess=nn.Linear(kwargs["steps_2"], topo_out_units), *args, **kwargs)
        self.fc = nn.Sequential(
                    nn.Linear(784 + 2*topo_out_units, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                    )

    def forward(self, x):
        x, x_dtm = x

        # first sig ECLayr
        ecc_1 = self.sig_eclayr_1(x_dtm)
        ecc_1 = F.relu(ecc_1)

        # CNN
        x = self.conv(x)

        # second sig ECLayr after conv layer
        min_vals = x.amin(dim=(2, 3), keepdim=True)     # shape: [B, C, 1, 1]
        max_vals = x.amax(dim=(2, 3), keepdim=True)     # shape: [B, C, 1, 1]
        x_2 = (x - min_vals) / (max_vals - min_vals)    # normalize between 0 and 1 for each data and channel
        ecc_2 = self.sig_eclayr_2(x_2)
        ecc_2 = F.relu(ecc_2)

        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, ecc_1, ecc_2), dim=-1)
        x = self.fc(x)
        return x