import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from eclayr.cubical.cubeclayr import ECLayr, SigECLayr
from utils.dtm import WeightedDTMLayer


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
        x, x_dtm005, x_dtm02 = x
        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Cnn + ECLayr
class EcCnn_i(Cnn):
    def __init__(self, in_channels=1, num_classes=10, gtheta_cfg=[32, 32], *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc = ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(784 + gtheta_cfg[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )

    def forward(self, x):
        x, x_dtm005, x_dtm02 = x
        
        # ECLayr
        x_1 = F.relu(self.ecc(x))
        
        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1), dim=-1)
        x = self.fc(x)
        return x


# Cnn + ECLayr + ECLayr after conv
class EcCnn(Cnn):
    def __init__(self, in_channels=1, num_classes=10, gtheta_cfg=[32, 32], *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc_1 = ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_2 = ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
                    nn.Linear(784 + 2*gtheta_cfg[-1], 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                    )

    def forward(self, x):
        x, x_dtm005, x_dtm02 = x

        # ECLayr 1
        x_1 = F.relu(self.ecc_1(x))

        x = self.conv(x)

        # ECLayr 2 after first conv layer
        x_2 = (x - x.min().item()) / (x.max().item() - x.min().item())  # normalize x_2 between 0 and 1
        x_2 = F.relu(self.ecc_2(x_2))

        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1, x_2), dim=-1)
        x = self.fc(x)
        return x


class EcCnnDTM_i(Cnn):
    def __init__(self, in_channels=1, num_classes=10, gtheta_cfg=[32, 32],
                 *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc_1 = ECLayr(interval=kwargs["interval_one"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_2 = ECLayr(interval=kwargs["interval_two"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(784 + 2*gtheta_cfg[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )

    def forward(self, x):
        x, x_dtm005, x_dtm02 = x

        # ECLayr 1
        x_1 = F.relu(self.ecc_1(x_dtm005))
        
        # ECLayr 2
        x_2 = F.relu(self.ecc_2(x_dtm02))

        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1, x_2), dim=-1)
        x = self.fc(x)
        return x


class EcCnnDTM(Cnn):
    def __init__(self, in_channels=1, num_classes=10, gtheta_cfg=[32, 32],
                 *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc_1 = ECLayr(interval=kwargs["interval_one"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_2 = ECLayr(interval=kwargs["interval_two"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_3 = ECLayr(interval=kwargs["interval_one"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(784 + 3*gtheta_cfg[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )
        self.dtm = WeightedDTMLayer(m0=0.05, size=kwargs["size"])

    def forward(self, x):
        x, x_dtm005, x_dtm02 = x

        # ECLayr 1
        x_1 = F.relu(self.ecc_1(x_dtm005))
        
        # ECLayr 2
        x_2 = F.relu(self.ecc_2(x_dtm02))

        x = self.conv(x)

        # ECLayr 3 after first conv layer
        x_3 = (x - x.min().item()) / (x.max().item() - x.min().item())  # normalize x_3 between 0 and 1
        x_3 = self.dtm(x_3)
        x_3 = F.relu(self.ecc_3(x_3))

        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x


class SigEcCnn(Cnn):
    def __init__(self, in_channels=1, num_classes=10, gtheta_cfg=[32, 32], *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc_1 = SigECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_2 = SigECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
                    nn.Linear(784 + 2*gtheta_cfg[-1], 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                    )

    def forward(self, x):
        x, x_dtm005, x_dtm02 = x

        # ECLayr 1
        x_1 = F.relu(self.ecc_1(x))

        x = self.conv(x)

        # ECLayr 2 after first conv layer
        x_2 = (x - x.min().item()) / (x.max().item() - x.min().item())  # normalize x_2 between 0 and 1
        x_2 = F.relu(self.ecc_2(x_2))

        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1, x_2), dim=-1)
        x = self.fc(x)
        return x


class SigEcCnnDTM(Cnn):
    def __init__(self, in_channels=1, num_classes=10, gtheta_cfg=[32, 32],
                 *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc_1 = SigECLayr(interval=kwargs["interval_one"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_2 = SigECLayr(interval=kwargs["interval_two"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_3 = SigECLayr(interval=kwargs["interval_one"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(784 + 3*gtheta_cfg[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )
        self.dtm = WeightedDTMLayer(m0=0.05, size=kwargs["size"])

    def forward(self, x):
        x, x_dtm005, x_dtm02 = x

        # ECLayr 1
        x_1 = F.relu(self.ecc_1(x_dtm005))
        
        # ECLayr 2
        x_2 = F.relu(self.ecc_2(x_dtm02))

        x = self.conv(x)

        # ECLayr 3 after first conv layer
        x_3 = (x - x.min().item()) / (x.max().item() - x.min().item())  # normalize x_3 between 0 and 1
        x_3 = self.dtm(x_3)
        x_3 = F.relu(self.ecc_3(x_3))

        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x