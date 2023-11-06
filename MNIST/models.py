import torch
import torch.nn as nn
import numpy as np
from pllay import TopoWeightLayer
from pllay_adap import AdTopoLayer


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

        self.in_channels = 64   # channel of input that goes into res_layer1

        self.conv_layer = nn.Sequential(nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.in_channels),
                                        nn.ReLU())
        
        self.res_layer_1 = self._make_layers(block, 64, cfg[0], stride=1)
        self.res_layer_2 = self._make_layers(block, 128, cfg[1], stride=2)
        # self.res_layer_3 = self._make_layers(block, 256, cfg[2], stride=2)
        # self.res_layer_4 = self._make_layers(block, 512, cfg[3], stride=2)

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        # weight initialization
        num_res_blocks = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, ResidualBlock):
                num_res_blocks += 1

        self.num_res_layers = (np.cumsum(cfg) == num_res_blocks).nonzero()[0].item() + 1
        self.fc = nn.Linear((2**(5+self.num_res_layers)) * block.expansion, num_classes)

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
        # x = self.res_layer_3(x)
        # x = self.res_layer_4(x)
        x = self.pool(x)
        ####################################################
        signal = torch.abs(x.detach()).sum(dim=0)
        ####################################################
        output = self.fc(x)
        return output, signal


class AdPllayResNet(ResNet):
    def __init__(self, block, cfg, out_features=50, num_classes=10):
        super().__init__(block, cfg, num_classes)
        self.topo_layer_1 = nn.Sequential(nn.Flatten(),
                                        AdTopoLayer(out_features, T=25, m0=0.05, K_max=2, lims=[[27, 0], [0, 27]], robust=True),   # hyperparameter 수정
                                        nn.BatchNorm1d(out_features),
                                        nn.ReLU())
        
        self.topo_layer_2 = nn.Sequential(nn.Flatten(),
                                         AdTopoLayer(out_features, T=25, m0=0.2, K_max=2, lims=[[27, 0], [0, 27]], robust=True),   # hyperparameter 수정
                                         nn.BatchNorm1d(out_features),
                                         nn.ReLU())
        
        self.num_topo_layers = 0
        for m in self.modules():
            if isinstance(m, AdTopoLayer):
                self.num_topo_layers += 1
            elif isinstance(m, nn.BatchNorm2d): # initialize BN of topo_layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.fc = nn.Linear((2**(5+self.num_res_layers))*block.expansion + self.num_topo_layers*out_features, num_classes)
    
    def forward(self, input):
        x = self.conv_layer(input)

        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        # x = self.res_layer_3(x)
        # x = self.res_layer_4(x)
        x_0 = self.pool(x)

        x_1 = self.topo_layer_1(input)
        if self.num_topo_layers == 2:
            x_2 = self.topo_layer_2(input)
            x_3 = torch.concat((x_0, x_1, x_2), dim=-1)
        else:
            x_3 = torch.concat((x_0, x_1), dim=-1)

        signal = torch.abs(x_3.detach()).sum(dim=0)
        output = self.fc(x_3)
        return output, signal

    def load_pretrained_pllay(self, pllay_pretrained_file_1, pllay_pretrained_file_2=None, freeze=False):
        model_dict = self.state_dict()
        
        pllay_pretrained_1 = torch.load(pllay_pretrained_file_1)
        pllay_params_1 = {layer:params for layer, params in pllay_pretrained_1.items() if layer in model_dict and "fc" not in layer}
        model_dict.update(pllay_params_1)

        if not (pllay_pretrained_file_2 is None):
            pllay_pretrained_2 = torch.load(pllay_pretrained_file_2)
            pllay_params_2 = {layer:params for layer, params in pllay_pretrained_2.items() if layer in model_dict and "fc" not in layer}
            model_dict.update(pllay_params_2)
        
        self.load_state_dict(model_dict)

        # only pllay network is pretrained and freezed
        if freeze:
            for m in self.modules():
                if isinstance(m, AdTopoLayer):
                    m.requires_grad_(False)


class ResNet18(ResNet):
    def __init__(self, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10):
        super().__init__(block, cfg, num_classes)


class ResNet34(ResNet):
    def __init__(self, block=ResidualBlock, cfg=[3,4,6,3], num_classes=50):
        super().__init__(block, cfg, num_classes)


class AdPRNet18(AdPllayResNet):
    def __init__(self, block=ResidualBlock, cfg=[2,2,2,2], out_features=50, num_classes=10):
        super().__init__(block, cfg, out_features, num_classes)