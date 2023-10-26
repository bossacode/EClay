import torch
import torch.nn as nn
import numpy as np
from pllay import TopoWeightLayer
from pllay_adap import AdaptiveTopoWeightLayer


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU())
        self.conv_layer2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv_layer1(input)
        x = self.conv_layer2(x)
        x = x + input if self.downsample is None else x + self.downsample(input)
        output = self.relu(x)
        return output


class ResNet(nn.Module):
    def __init__(self, block, cfg, num_classes=10):
        super().__init__()
        # change architecture of ResNet bc. our image size (3, 32, 32) is too small for the original architecture
        # self.in_channels = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.bn = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()

        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 3   # channel of input that goes into res_layer1

        self.conv_layer = nn.Sequential(nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.in_channels),
                                        nn.ReLU())
        
        self.res_layer_1 = self._make_layers(block, 64, cfg[0], stride=1)
        self.res_layer_2 = self._make_layers(block, 128, cfg[1], stride=2)
        self.res_layer_3 = self._make_layers(block, 256, cfg[2], stride=2)
        self.res_layer_4 = self._make_layers(block, 512, cfg[3], stride=2)

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, first_conv_channel, num_blocks, stride):      
        if stride != 1 or self.in_channels != first_conv_channel * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, first_conv_channel*block.expansion, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(first_conv_channel, block.expansion))
        else:
            downsample = None
        
        block_list =[]
        block_list.append(block(self.in_channels, first_conv_channel, stride, downsample))

        self.in_channels = first_conv_channel * block.expansion
        
        for _ in range(1, num_blocks):
            block_list.append(block(self.in_channels, first_conv_channel))
        return nn.Sequential(*block_list)

    def forward(self, input):
        x = self.conv_layer(input)
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x = self.pool(x)
        output = self.fc(x)
        return output


class PllayResNet(ResNet):
    def __init__(self, block, cfg, out_features=32, num_classes=10):
        super().__init__(block, cfg, num_classes)
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        TopoWeightLayer(out_features, tseq=np.linspace(0.06, 0.3, 25), m0=0.05, K_max=2),
                                        nn.ReLU())    # hyperparameter 수정
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        TopoWeightLayer(out_features, tseq=np.linspace(0.14, 0.4, 27), m0=0.2, K_max=3),
                                        nn.ReLU())     # hyperparameter 수정
        
        self.fc = nn.Linear(512*block.expansion + 2*out_features, num_classes)
    
    def forward(self, input):
        x = self.conv_layer(input)
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x_1 = self.pool(x)

        x_2 = self.topo_layer_1(input)
        x_3 = self.topo_layer_2(input)

        output = self.fc(torch.concat((x_1, x_2, x_3), dim=-1))
        return output


class AdaptivePllayResNet(ResNet):
    def __init__(self, block, cfg, out_features=32, num_classes=10):
        super().__init__(block, cfg, num_classes)
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        AdaptiveTopoWeightLayer(out_features, T=50, m0=0.05, K_max=2),  # hyperparameter 수정
                                        nn.ReLU())
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                        AdaptiveTopoWeightLayer(out_features, T=50, m0=0.05, K_max=2),   # hyperparameter 수정
                                        nn.ReLU())
        self.topo_layer_3 = nn.Sequential(nn.Flatten(),
                                AdaptiveTopoWeightLayer(out_features, T=50, m0=0.05, K_max=2),   # hyperparameter 수정
                                nn.ReLU())
        self.topo_layer_4 = nn.Sequential(nn.Flatten(),
                                AdaptiveTopoWeightLayer(out_features, T=50, m0=0.05, K_max=2),   # hyperparameter 수정
                                nn.ReLU())
        
        self.fc = nn.Linear(512*block.expansion + 4*out_features, num_classes)
    
    def forward(self, input):
        x = self.conv_layer(input)

        x_0 = self.topo_layer_1(x[:,0,:,:])
        x_1 = self.topo_layer_1(x[:,1,:,:])
        x_2 = self.topo_layer_1(x[:,2,:,:])

        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x = self.pool(x)

        x_3 = self.topo_layer_4(input)

        output = self.fc(torch.concat((x, x_0, x_1, x_2, x_3), dim=-1))
        return output


class ResNet18(ResNet):
    def __init__(self, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10):
        super().__init__(block, cfg, num_classes)


class ResNet34(ResNet):
    def __init__(self, block=ResidualBlock, cfg=[3,4,6,3], num_classes=10):
        super().__init__(block, cfg, num_classes)


class PRNet18(PllayResNet):
    def __init__(self, block=ResidualBlock, cfg=[2,2,2,2], out_features=32, num_classes=10):
        super().__init__(block, cfg, out_features, num_classes)


class PRNet34(PllayResNet):
    def __init__(self, block=ResidualBlock, cfg=[3,4,6,3], out_features=32, num_classes=10):
        super().__init__(block, cfg, out_features, num_classes)


class AdaptivePRNet18(AdaptivePllayResNet):
    def __init__(self, block=ResidualBlock, cfg=[2,2,2,2], out_features=32, num_classes=10):
        super().__init__(block, cfg, out_features, num_classes)