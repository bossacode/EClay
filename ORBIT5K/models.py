import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from eclayr.cubical import cubeclayr
from eclayr.alpha import alphaeclayr
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
            nn.Linear(1600, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_pc, x, x_dtm = x
        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Cnn + ECLayr
class EcCnn_i(Cnn):
    def __init__(self, in_channels=1, num_classes=5, gtheta_cfg=[32, 32], *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc = alphaeclayr.ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1600 + gtheta_cfg[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )

    def forward(self, x):
        x_pc, x, x_dtm = x
        
        # ECLayr
        x_1 = F.relu(self.ecc(x_pc))
        
        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1), dim=-1)
        x = self.fc(x)
        return x


# Cnn + ECLayr + ECLayr after conv
class EcCnn(Cnn):
    def __init__(self, in_channels=1, num_classes=5, gtheta_cfg=[32, 32], *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc_1 = alphaeclayr.ECLayr(interval=kwargs["interval_one"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_2 = cubeclayr.ECLayr(interval=kwargs["interval_two"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
                    nn.Linear(1600 + 2*gtheta_cfg[-1], 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                    )

    def forward(self, x):
        x_pc, x, x_dtm = x

        # ECLayr 1
        x_1 = F.relu(self.ecc_1(x_pc))

        # Cnn
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
    def __init__(self, in_channels=1, num_classes=5, gtheta_cfg=[32, 32],
                 *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc = cubeclayr.ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1600 + gtheta_cfg[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )

    def forward(self, x):
        x_pc, x, x_dtm = x

        # ECLayr 1
        x_1 = F.relu(self.ecc(x_dtm))

        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1), dim=-1)
        x = self.fc(x)
        return x


class EcCnnDTM(Cnn):
    def __init__(self, in_channels=1, num_classes=5, gtheta_cfg=[32, 32],
                 *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.ecc_1 = cubeclayr.ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_2 = cubeclayr.ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1600 + 2*gtheta_cfg[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )
        self.dtm = WeightedDTMLayer(m0=0.01, size=kwargs["size"])

    def forward(self, x):
        x_pc, x, x_dtm = x

        # ECLayr 1
        x_1 = F.relu(self.ecc_1(x_dtm))
        
        # Cnn
        x = self.conv(x)

        # ECLayr 3 after first conv layer
        x_2 = (x - x.min().item()) / (x.max().item() - x.min().item())  # normalize x_3 between 0 and 1
        x_2 = self.dtm(x_2)
        x_2 = F.relu(self.ecc_2(x_2))

        x = F.relu(x)
        x = self.flatten(x)

        x = torch.concat((x, x_1, x_2), dim=-1)
        x = self.fc(x)
        return x















# # ResNet
# class ResidualBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super().__init__()
#         self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
#                                         nn.BatchNorm2d(out_channels))
#         self.conv_layer2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                                         nn.BatchNorm2d(out_channels))
#         self.downsample = downsample
#         self.relu = nn.ReLU()

#     def forward(self, input):
#         x = self.conv_layer1(input)
#         x = self.relu(x)
#         x = self.conv_layer2(x)
#         x = x + input if self.downsample is None else x + self.downsample(input)
#         return x


# class ResNet18(nn.Module):
#     def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock):
#         """
#         Args:
#             in_channels (int): Number of channels of input data. Defaults to 1.
#             num_classes (int, optional): Number of classes. Defaults to 10.
#             block_cfg (list, optional): Number of blocks for each residual layer. Defaults to [2, 2, 2, 2].
#             filter_cfg (list, optional): Number of filters at the start of each residual layer. Defaults to [64, 128, 256, 512].
#             block (_type_, optional): Type of block to use. Defaults to ResidualBlock.
#         """
#         super().__init__()
#         assert len(block_cfg) == len(filter_cfg) == 4
#         self.res_in_channels = in_channels  # channel of input that goes into res_layer1, value changes in _make_layers

#         self.res_layer_1 = self._make_layers(block, filter_cfg[0], block_cfg[0], stride=1)
#         self.res_layer_2 = self._make_layers(block, filter_cfg[1], block_cfg[1], stride=2)
#         self.res_layer_3 = self._make_layers(block, filter_cfg[2], block_cfg[2], stride=2)
#         self.res_layer_4 = self._make_layers(block, filter_cfg[3], block_cfg[3], stride=2)
#         self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
#         self.fc = nn.Sequential(nn.Linear(filter_cfg[-1], 64),
#                                 nn.ReLU(),
#                                 nn.Linear(64, num_classes))
#         self.relu = nn.ReLU()

#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x, x_dtm005, x_dtm02 = x
#         x = self.relu(self.res_layer_1(x))
#         x = self.relu(self.res_layer_2(x))
#         x = self.relu(self.res_layer_3(x))
#         x = self.relu(self.res_layer_4(x))
#         x = self.avg_pool(x)
#         output = self.fc(x)
#         return output

#     def _make_layers(self, block, num_filters, num_blocks, stride):
#         """
#         Args:
#             block: Type of block to use. ResidualBlock or BottleneckBlock
#             num_filters: Number of filters at the start of each layer
#             num_blocks: Number of blocks for each layer
#             stride:
#         """
#         if stride != 1 or self.res_in_channels != num_filters * block.expansion:
#             downsample = nn.Sequential(nn.Conv2d(self.res_in_channels, num_filters*block.expansion, kernel_size=1, stride=stride),
#                                     nn.BatchNorm2d(num_filters, block.expansion))
#         else:
#             downsample = None
        
#         block_list =[]
#         block_list.append(block(self.res_in_channels, num_filters, stride, downsample))

#         self.res_in_channels = num_filters * block.expansion
        
#         for _ in range(1, num_blocks):
#             block_list.append(nn.ReLU())
#             block_list.append(block(self.res_in_channels, num_filters))
#         return nn.Sequential(*block_list)


# # ResNet + ECLayr
# class EcResNet_i(ResNet18):
#     def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,  # ResNet params
#                  gtheta_cfg=[32, 64, 32], *args, **kwargs):
#         super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
#         self.ecc = ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + gtheta_cfg[-1], 64),
#                         nn.ReLU(),
#                         nn.Linear(64, num_classes))
    
#     def forward(self, x):
#         x, ecc_dtm005, ecc_dtm02 = x
        
#         x_1 = self.relu(self.ecc(x))        # ECLayr 1

#         x = self.relu(self.res_layer_1(x))  # ResNet
#         x = self.relu(self.res_layer_2(x))
#         x = self.relu(self.res_layer_3(x))
#         x = self.relu(self.res_layer_4(x))
#         x_2 = self.avg_pool(x)

#         x = torch.concat((x_1, x_2), dim=-1)
#         x = self.fc(x)
#         return x


# # Cnn + ECLayr + ECLayr after Residual Layers
# class EcResNet(ResNet18):
#     def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,      # ResNet params
#                  size_1=[28, 28],
#                  size_2=[14, 14],
#                  gtheta_cfg=[32, 64, 32], *args, **kwargs):
#         super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
#         self.ecc_1 = ECLayr(size=size_1, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.ecc_2 = ECLayr(size=size_1, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.ecc_3 = ECLayr(size=size_2, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.bn_1 = nn.BatchNorm2d(1)
#         self.bn_2 = nn.BatchNorm2d(1)
#         self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 3*gtheta_cfg[-1], 64),
#                         nn.ReLU(),
#                         nn.Linear(64, num_classes))
    
#     def forward(self, x):
#         x, x_dtm005, x_dtm02 = x
        
#         x_1 = self.relu(self.ecc_1(x))      # ECLayr 1

#         x = self.res_layer_1(x)             # ResNet

#         # insert ECLay after first res layer
#         x_2 = x.mean(dim=1, keepdim=True)
#         x_2 = self.bn_1(x_2)
#         x_2 = (x_2 - x_2.min()) / (x_2.max() - x_2.min())   # normalize x_1 between 0 and 1
#         x_2 = self.relu(self.ecc_2(x_2))    # ECLayr 2

#         x = self.res_layer_2(self.relu(x))

#         # insert ECLay after second res layer
#         x_3 = x.mean(dim=1, keepdim=True)
#         x_3 = self.bn_2(x_3)
#         x_3 = (x_3 - x_3.min()) / (x_3.max() - x_3.min())   # normalize x_1 between 0 and 1
#         x_3 = self.relu(self.ecc_3(x_3))    # ECLayr 3

#         x = self.relu(self.res_layer_3(self.relu(x)))
#         x = self.relu(self.res_layer_4(x))
#         x_4 = self.avg_pool(x)
        
#         x = torch.concat((x_1, x_2, x_3, x_4), dim=-1)
#         x = self.fc(x)
#         return x


# # ResNet + DTM ECLayr
# class EcResNetDTM_i(ResNet18):
#     def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,  # ResNet params
#                  interval_1 = [0.03, 0.34],
#                  interval_2 = [0.06, 0.35],
#                  gtheta_cfg=[32, 64, 32], *args, **kwargs):
#         super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
#         self.ecc_1 = ECLayr(interval=interval_1, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.ecc_2 = ECLayr(interval=interval_2, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 2*gtheta_cfg[-1], 64),
#                         nn.ReLU(),
#                         nn.Linear(64, num_classes))
    
#     def forward(self, x):
#         x, x_dtm005, x_dtm02 = x
        
#         x_1 = self.relu(self.ecc_1(x_dtm005))   # ECLayr 1
#         x_2 = self.relu(self.ecc_2(x_dtm02))    # ECLayr 2

#         x = self.relu(self.res_layer_1(x))  # ResNet
#         x = self.relu(self.res_layer_2(x))
#         x = self.relu(self.res_layer_3(x))
#         x = self.relu(self.res_layer_4(x))
#         x_3 = self.avg_pool(x)

#         x = torch.concat((x_1, x_2, x_3), dim=-1)
#         x = self.fc(x)
#         return x


# class EcResNetDTM(ResNet18):
#     def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,      # ResNet params
#                  sublevel_1=True, size_1=[28, 28], interval_1 = [0.03, 0.34],   # DTM 1
#                  interval_2 = [0.06, 0.35],                                     # DTM 2
#                  sublevel_2 = False, interval_3 = [0, 1],                       # after layer 1
#                  size_2=[14, 14],                                               # after layer 2
#                  gtheta_cfg=[32, 64, 32], *args, **kwargs):
#         super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
#         self.ecc_1 = ECLayr(sublevel=sublevel_1, size=size_1, interval=interval_1, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.ecc_2 = ECLayr(sublevel=sublevel_1, size=size_1, interval=interval_2, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.ecc_3 = ECLayr(sublevel=sublevel_2, size=size_1, interval=interval_3, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.ecc_4 = ECLayr(sublevel=sublevel_2, size=size_2, interval=interval_3, gtheta_cfg=gtheta_cfg, *args, **kwargs)
#         self.bn_1 = nn.BatchNorm2d(1)
#         self.bn_2 = nn.BatchNorm2d(1)
#         self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 4*gtheta_cfg[-1], 64),
#                         nn.ReLU(),
#                         nn.Linear(64, num_classes))
    
#     def forward(self, x):
#         x, x_dtm005, x_dtm02 = x
        
#         x_1 = self.relu(self.ecc_1(x_dtm005))   # ECLayr 1
#         x_2 = self.relu(self.ecc_2(x_dtm02))    # ECLayr 2

#         x = self.res_layer_1(x)             # ResNet

#         # insert ECLay after first res layer
#         x_3 = x.mean(dim=1, keepdim=True)
#         x_3 = self.bn_1(x_3)
#         x_3 = (x_3 - x_3.min()) / (x_3.max() - x_3.min())   # normalize x_1 between 0 and 1
#         x_3 = self.relu(self.ecc_3(x_3))    # ECLayr 2

#         x = self.res_layer_2(self.relu(x))

#         # insert ECLay after second res layer
#         x_4 = x.mean(dim=1, keepdim=True)
#         x_4 = self.bn_2(x_4)
#         x_4 = (x_4 - x_4.min()) / (x_4.max() - x_4.min())   # normalize x_1 between 0 and 1
#         x_4 = self.relu(self.ecc_4(x_4))    # ECLayr 3

#         x = self.relu(self.res_layer_3(self.relu(x)))
#         x = self.relu(self.res_layer_4(x))
#         x_5 = self.avg_pool(x)
        
#         x = torch.concat((x_1, x_2, x_3, x_4, x_5), dim=-1)
#         x = self.fc(x)
#         return x















# class PLResNet(ResNet18):
#     def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,  # ResNet params
#                  hidden_features=[64, 32]):
#         super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
#         self.gtheta_1 = GThetaPL(num_features=[128] + hidden_features)
#         self.gtheta_2 = GThetaPL(num_features=[128] + hidden_features)
#         self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 2*hidden_features[-1], 64),
#                                 nn.ReLU(),
#                                 nn.Linear(64, num_classes))
#         self.flatten = nn.Flatten()

    
#     def forward(self, x):
#         x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
#         # ResNet
#         x = self.res_layer_1(x)
#         x = self.res_layer_2(x)
#         x = self.res_layer_3(x)
#         x = self.res_layer_4(x)
#         x_1 = self.avg_pool(x)
#         # PL layer 1
#         x_2 = self.gtheta_1(self.flatten(pl_dtm005))
#         # PL layer 2
#         x_3 = self.gtheta_2(self.flatten(pl_dtm02))
#         # FC layer
#         x = torch.concat((x_1, x_2, x_3), dim=-1)
#         x = self.fc(x)
#         return x

# class PLResNet_Topo(ResNet18):
#     def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,          # ResNet params
#                  as_vertices=False, sublevel=False, interval=[-2.5, 0], steps=32, K_max=2, dimensions=[0, 1], hidden_features=[64, 32]):    # PL params
#         super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
#         self.gtheta_1 = GThetaPL(num_features=[128] + hidden_features)
#         self.gtheta_2 = GThetaPL(num_features=[128] + hidden_features)
#         self.topo_layer_3 = PLLay(as_vertices, sublevel, interval, steps, K_max, dimensions, in_channels, hidden_features)
#         self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 3*hidden_features[-1], 64),
#                                 nn.ReLU(),
#                                 nn.Linear(64, num_classes))
#         self.flatten = nn.Flatten()
#         self.dtm = WeightedDTMLayer(m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
    
#     def forward(self, x):
#         x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
#         # ResNet
#         x = self.res_layer_1(x)

#         # insert PLLay after first res layer
#         x_2 = x.mean(dim=1, keepdim=True)
#         # if ((x_2 == 0).sum(dim=(2,3)) / (x.shape[2]*x.shape[3]) >= 0.90).any(): # to avoid dtm grad resulting in nan when input is very sparse
#         #     print("no dtm", x_2.min().item(), x_2.max().item())
#         #     x_2 = self.topo_layer_3(x_2)
#         # else:
#         #     x_2 = self.dtm(x_2)
#         #     print("dtm", x_2.min().item(), x_2.max().item())
#         #     x_2 = self.topo_layer_3(x_2)
#         x_2 = self.dtm(x_2)
#         x_2 = self.topo_layer_3(x_2)

#         x = self.res_layer_2(x)
#         x = self.res_layer_3(x)
#         x = self.res_layer_4(x)
#         x_1 = self.avg_pool(x)
#         # Pl layer 1
#         x_3 = self.gtheta_1(self.flatten(pl_dtm005))
#         # PL layer 2
#         x_4 = self.gtheta_2(self.flatten(pl_dtm02))
#         # FC layer
#         x = torch.concat((x_1, x_2, x_3, x_4), dim=-1)
#         x = self.fc(x)
#         return x
    

# class EctResNet(ResNet18):
#     def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,  # ResNet params
#                  bump_steps=32, num_features=3, num_thetas=32, R=1.1, ect_type="faces", device="cpu", fixed=False):                 # ECT params
#         super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
#         self.ectlayer = EctLayer(bump_steps, num_features, num_thetas, R, ect_type, device, fixed)

#     def forward(self, x):
#         x = self.ectlayer(x).unsqueeze(1)
#         x = self.res_layer_1(x)
#         x = self.res_layer_2(x)
#         x = self.res_layer_3(x)
#         x = self.res_layer_4(x)
#         x = self.avg_pool(x)
#         output = self.fc(x)
#         return output


# class PLCNN(CNN):
#     def __init__(self, in_channels=1, num_classes=10, hidden_features=[32]):
#         super().__init__(in_channels, num_classes)
#         self.gtheta_1 = GThetaPL(num_features=[128] + hidden_features)
#         self.gtheta_2 = GThetaPL(num_features=[192] + hidden_features)
#         self.fc = nn.Sequential(nn.Linear(784 + 2*hidden_features[-1], 64),
#                                 nn.ReLU(),
#                                 nn.Linear(64, num_classes))
    
#     def forward(self, x):
#         x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
#         # CNN
#         x_1 = self.flatten(self.conv_layer(x))
#         # PL layer 1
#         x_2 = self.gtheta_1(self.flatten(pl_dtm005))
#         # PL layer 2
#         x_3 = self.gtheta_2(self.flatten(pl_dtm02))
#         # FC layer
#         x = torch.concat((x_1, x_2, x_3), dim=-1)
#         x = self.fc(x)
#         return x


# class PLCNN_Topo(CNN):
#     def __init__(self, in_channels=1, num_classes=10,                                                                               # CNN params
#                  as_vertices=False, sublevel=False, interval=[-7, 0], steps=32, K_max=2, dimensions=[0, 1], hidden_features=[32]):  # EC params
#         super().__init__(in_channels, num_classes)
#         self.gtheta_1 = GThetaPL(num_features=[128] + hidden_features)
#         self.gtheta_2 = GThetaPL(num_features=[192] + hidden_features)
#         self.topo_layer_3 = PLLay(as_vertices, sublevel, interval, steps, K_max, dimensions, in_channels, hidden_features)
#         self.fc = nn.Sequential(nn.Linear(784 + 3*hidden_features[-1], 64),
#                                 nn.ReLU(),
#                                 nn.Linear(64, num_classes))
#         self.dtm = WeightedDTMLayer(m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

#     def forward(self, x):
#         x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
#         # CNN
#         x_1 = self.conv_layer(x)
#         # PL layer 1
#         x_2 = self.gtheta_1(self.flatten(pl_dtm005))
#         # PL layer 2
#         x_3 = self.gtheta_2(self.flatten(pl_dtm02))
#         # PL Layer 3
#         if ((x_1 == 0).sum(dim=(2,3)) / (x.shape[2]*x.shape[3]) >= 0.5).any(): # to avoid dtm grad resulting in nan when input is very sparse
#             x_4 = self.topo_layer_3(x_1)
#         else:
#             x_4 = self.dtm(x_1)
#             x_4 = self.topo_layer_3(x_4)

#         # FC layer
#         x = torch.concat((self.flatten(x_1), x_2, x_3, x_4), dim=-1)
#         x = self.fc(x)
#         return x


# class EctCnnModel(CNN):
#     def __init__(self, in_channels=1, num_classes=32,
#                  bump_steps=32, num_features=3, num_thetas=32, R=1.1, ect_type="faces", device="cpu", fixed=False):
#         super().__init__(in_channels, num_classes)
#         self.ectlayer = EctLayer(bump_steps, num_features, num_thetas, R, ect_type, device, fixed)
#         self.fc = nn.Sequential(nn.Linear(1024, 64),
#                                 nn.ReLU(),
#                                 nn.Linear(64, num_classes))

#     def forward(self, batch):
#         x = self.ectlayer(batch).unsqueeze(1)
#         x = self.conv_layer(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x