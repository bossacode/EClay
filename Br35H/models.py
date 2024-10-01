import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from eclayr.cubical.cubeclayr import ECLayr


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
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock):
        """
        Args:
            in_channels (int): Number of channels of input data. Defaults to 1.
            num_classes (int, optional): Number of classes. Defaults to 10.
            block_cfg (list, optional): Number of blocks for each residual layer. Defaults to [2, 2, 2, 2].
            filter_cfg (list, optional): Number of filters at the start of each residual layer. Defaults to [64, 128, 256, 512].
            block (_type_, optional): Type of block to use. Defaults to ResidualBlock.
        """
        super().__init__()
        assert len(block_cfg) == len(filter_cfg)
        self.res_in_channels = 64  # channel of input that goes into res_layer1, value changes in _make_layers
        
        self.conv = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_layer = nn.Sequential(
            self._make_layers(block, filter_cfg[0], block_cfg[0], stride=1),
            nn.ReLU(),
            self._make_layers(block, filter_cfg[1], block_cfg[1], stride=2),
            nn.ReLU(),
            self._make_layers(block, filter_cfg[2], block_cfg[2], stride=2),
            nn.ReLU(),
            self._make_layers(block, filter_cfg[3], block_cfg[3], stride=2),
            nn.ReLU()
        )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(filter_cfg[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))
        self.relu = nn.ReLU()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res_layer(x)
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
            block_list.append(nn.ReLU())
            block_list.append(block(self.res_in_channels, num_filters))
        return nn.Sequential(*block_list)


# ResNet + ECLayr
class EcResNet_i(ResNet):
    def __init__(self, in_channels=1, num_classes=10, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,  # ResNet params
                 gtheta_cfg=[32, 64, 32], *args, **kwargs):
        super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
        self.ecc = ECLayr(gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + gtheta_cfg[-1], 64),
                        nn.ReLU(),
                        nn.Linear(64, num_classes))
    
    def forward(self, x):        
        x_1 = self.relu(self.ecc(x))    # ECLayr 1

        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res_layer(x)
        x = self.avg_pool(x)

        x = torch.concat((x, x_1), dim=-1)
        x = self.fc(x)
        return x


# Cnn + ECLayr + ECLayr after Residual Layers
class EcResNet(ResNet):
    def __init__(self, in_channels=1, num_classes=2, block_cfg=[2, 2, 2, 2], filter_cfg=[64, 128, 256, 512], block=ResidualBlock,      # ResNet params
                 gtheta_cfg=[32, 64, 32], *args, **kwargs):
        super().__init__(in_channels, num_classes, block_cfg, filter_cfg, block)
        print(kwargs)
        self.ecc_1 = ECLayr(size=kwargs["size_one"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.ecc_2 = ECLayr(size=kwargs["size_two"], gtheta_cfg=gtheta_cfg, *args, **kwargs)
        self.fc = nn.Sequential(nn.Linear(filter_cfg[-1] + 2*gtheta_cfg[-1], 64),
                        nn.ReLU(),
                        nn.Linear(64, num_classes))
    
    def forward(self, x):
        x_1 = self.relu(self.ecc_1(x))      # ECLayr 1

        x = self.conv(x)
        x = self.max_pool(x)

        print(x.shape)

        # insert ECLay after first res layer
        x_2 = x.mean(dim=1, keepdim=True)
        x_2 = (x_2 - x_2.min().item()) / (x_2.max().item() - x_2.min().item())   # normalize x_1 between 0 and 1
        x_2 = self.relu(self.ecc_2(x_2))    # ECLayr 2

        x = self.res_layer(x)
        x = self.avg_pool(x)
        
        x = torch.concat((x, x_1, x_2), dim=-1)
        x = self.fc(x)
        return x