import torch
import torch.nn as nn
from dtm import WeightedDTMLayer
from eclay import ECLay, GThetaEC
from pllay import PLLay, GThetaPL
from dect import EctLayer


# ResNet
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                        nn.BatchNorm2d(out_channels))
        self.conv_layer2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv_layer1(input)
        x = self.relu(x)
        x = self.conv_layer2(x)
        x = x + input if self.downsample is None else x + self.downsample(input)
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock):
        """
        Args:
            in_channels (int): Number of channels of input data. Defaults to 1.
            num_classes (int, optional): Number of classes. Defaults to 10.
            block_cfg (list, optional): Number of blocks for each residual layer. Defaults to [2, 2, 2, 2].
            filter_cfg (list, optional): Number of filters at the start of each residual layer. Defaults to [64, 128, 256, 512].
            block (_type_, optional): Type of block to use. Defaults to ResidualBlock.
        """
        super().__init__()
        assert len(block_cfg) == len(filter_cfg) == 4
        self.res_in_channels = 1   # channel of input that goes into res_layer1, value changes in _make_layers

        self.res_layer_1 = self._make_layers(block, filter_cfg[0], block_cfg[0], stride=1)
        self.res_layer_2 = self._make_layers(block, filter_cfg[1], block_cfg[1], stride=2)
        self.res_layer_3 = self._make_layers(block, filter_cfg[2], block_cfg[2], stride=2)
        self.res_layer_4 = self._make_layers(block, filter_cfg[3], block_cfg[3], stride=2)
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(filter_cfg[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x = self.avg_pool(x)
        output = self.fc(x)
        return output

    def _make_layers(self, block, num_filters, num_blocks, stride):
        """
        Args:
            block: Type of block to use. ResidualBlock or BottleneckBlock
            num_filters: Number of filters at the start of each layer
            num_blocks: Number of blocks for each layer
            stride:
        """
        if stride != 1 or self.res_in_channels != num_filters * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.res_in_channels, num_filters*block.expansion, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(num_filters, block.expansion))
        else:
            downsample = None
        
        block_list =[]
        block_list.append(block(self.res_in_channels, num_filters, stride, downsample))

        self.res_in_channels = num_filters * block.expansion
        
        for _ in range(1, num_blocks):
            block_list.append(block(self.res_in_channels, num_filters))
        return nn.Sequential(*block_list)


class ECResNet(ResNet18):
    def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,  # ResNet params
                 hidden_features=[64, 32]):
        super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
        self.gtheta_1 = GThetaEC(num_features=[32] + hidden_features)
        self.gtheta_2 = GThetaEC(num_features=[32] + hidden_features)
        self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 2*hidden_features[-1], 64),
                        nn.ReLU(),
                        nn.Linear(64, num_classes))
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        # ResNet
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x_1 = self.avg_pool(x)
        # ECC layer 1
        x_2 = self.gtheta_1(self.flatten(ecc_dtm005))
        # ECC layer 2
        x_3 = self.gtheta_2(self.flatten(ecc_dtm02))
        # FC layer
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x


class PLResNet(ResNet18):
    def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,  # ResNet params
                 hidden_features=[64, 32]):
        super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
        self.gtheta_1 = GThetaPL(num_features=[128] + hidden_features)
        self.gtheta_2 = GThetaPL(num_features=[128] + hidden_features)
        self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 2*hidden_features[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))
        self.flatten = nn.Flatten()

    
    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        # ResNet
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x_1 = self.avg_pool(x)
        # PL layer 1
        x_2 = self.gtheta_1(self.flatten(pl_dtm005))
        # PL layer 2
        x_3 = self.gtheta_2(self.flatten(pl_dtm02))
        # FC layer
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x


class ECResNet_Topo(ResNet18):
    def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,      # ResNet params
                 as_vertices=False, sublevel=False, size=[28, 28], interval=[-2.5, 0], steps=32, hidden_features=[64, 32], scale=0.1):  # EC params
        super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
        self.gtheta_1 = GThetaEC(num_features=[32] + hidden_features)
        self.gtheta_2 = GThetaEC(num_features=[32] + hidden_features)
        self.topo_layer_3 = ECLay(as_vertices, sublevel, size, interval, steps, in_channels, hidden_features, scale=scale)
        self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 3*hidden_features[-1], 64),
                        nn.ReLU(),
                        nn.Linear(64, num_classes))
        self.flatten = nn.Flatten()
        self.dtm = WeightedDTMLayer(m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
    
    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        # ResNet
        x = self.res_layer_1(x)

        # insert ECLay after first res layer
        x_2 = x.mean(dim=1, keepdim=True)
        # if ((x_2 == 0).sum(dim=(2,3)) / (x.shape[2]*x.shape[3]) >= 0.90).any(): # to avoid dtm grad resulting in nan when input is very sparse
        #     x_2 = self.topo_layer_3(x_2)
        # else:
        #     x_2 = self.dtm(x_2)
        #     x_2 = self.topo_layer_3(x_2)
        x_2 = self.dtm(x_2)
        x_2 = self.topo_layer_3(x_2)

        x = self.res_layer_2(x)        
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x_1 = self.avg_pool(x)
        # EC layer 1
        x_3 = self.gtheta_1(self.flatten(ecc_dtm005))
        # EC layer 2
        x_4 = self.gtheta_2(self.flatten(ecc_dtm02))
        # FC layer
        x = torch.concat((x_1, x_2, x_3, x_4), dim=-1)
        x = self.fc(x)
        return x
    

class PLResNet_Topo(ResNet18):
    def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,          # ResNet params
                 as_vertices=False, sublevel=False, interval=[-2.5, 0], steps=32, K_max=2, dimensions=[0, 1], hidden_features=[64, 32]):    # PL params
        super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
        self.gtheta_1 = GThetaPL(num_features=[128] + hidden_features)
        self.gtheta_2 = GThetaPL(num_features=[128] + hidden_features)
        self.topo_layer_3 = PLLay(as_vertices, sublevel, interval, steps, K_max, dimensions, in_channels, hidden_features)
        self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 3*hidden_features[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))
        self.flatten = nn.Flatten()
        self.dtm = WeightedDTMLayer(m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
    
    def forward(self, x):
        x, ecc_dtm005, ecc_dtm02, pl_dtm005, pl_dtm02 = x
        # ResNet
        x = self.res_layer_1(x)

        # insert PLLay after first res layer
        x_2 = x.mean(dim=1, keepdim=True)
        # if ((x_2 == 0).sum(dim=(2,3)) / (x.shape[2]*x.shape[3]) >= 0.90).any(): # to avoid dtm grad resulting in nan when input is very sparse
        #     print("no dtm", x_2.min().item(), x_2.max().item())
        #     x_2 = self.topo_layer_3(x_2)
        # else:
        #     x_2 = self.dtm(x_2)
        #     print("dtm", x_2.min().item(), x_2.max().item())
        #     x_2 = self.topo_layer_3(x_2)
        x_2 = self.dtm(x_2)
        x_2 = self.topo_layer_3(x_2)

        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x_1 = self.avg_pool(x)
        # Pl layer 1
        x_3 = self.gtheta_1(self.flatten(pl_dtm005))
        # PL layer 2
        x_4 = self.gtheta_2(self.flatten(pl_dtm02))
        # FC layer
        x = torch.concat((x_1, x_2, x_3, x_4), dim=-1)
        x = self.fc(x)
        return x
    

class EctResNet(ResNet18):
    def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,  # ResNet params
                 bump_steps=32, num_features=3, num_thetas=32, R=1.1, ect_type="faces", device="cpu", fixed=False):                 # ECT params
        super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
        self.ectlayer = EctLayer(bump_steps, num_features, num_thetas, R, ect_type, device, fixed)

    def forward(self, x):
        x = self.ectlayer(x).unsqueeze(1)
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x = self.avg_pool(x)
        output = self.fc(x)
        return output