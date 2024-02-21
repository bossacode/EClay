import torch
import torch.nn as nn
import numpy as np
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
    def __init__(self, in_channels, block, cfg, num_classes, res_in_channels, p):
        super().__init__()
        self.res_in_channels = res_in_channels   # channel of input that goes into res_layer1, value changes in _make_layers

        # original ResNet
        # self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, self.res_in_channels, kernel_size=7, stride=2, padding=3),
        #                                 nn.BatchNorm2d(self.res_in_channels),
        #                                 nn.ReLU())
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # modified ResNet
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, self.res_in_channels, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(self.res_in_channels),
                                nn.ReLU())
        
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.res_layer_1 = self._make_layers(block, res_in_channels, cfg[0], stride=1)
        self.res_layer_2 = self._make_layers(block, 2*res_in_channels, cfg[1], stride=2)
        self.res_layer_3 = self._make_layers(block, 4*res_in_channels, cfg[2], stride=2)
        self.res_layer_4 = self._make_layers(block, 8*res_in_channels, cfg[3], stride=2)

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(8*res_in_channels, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, first_conv_channel, num_blocks, stride):      
        if stride != 1 or self.res_in_channels != first_conv_channel * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.res_in_channels, first_conv_channel*block.expansion, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(first_conv_channel, block.expansion))
        else:
            downsample = None
        
        block_list =[]
        block_list.append(block(self.res_in_channels, first_conv_channel, stride, downsample))

        self.res_in_channels = first_conv_channel * block.expansion
        
        for _ in range(1, num_blocks):
            block_list.append(block(self.res_in_channels, first_conv_channel))
        return nn.Sequential(*block_list)

    def forward(self, input):
        x = self.conv_layer(input)
        # x = self.max_pool(x)
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x = self.pool(x)
        x = self.dropout(x)
        output = self.fc(x)
        return output


class AdPllayResNet(ResNet):
    def __init__(self, in_channels, block, cfg, out_features=50, num_classes=7, m0_1=0.05, m0_2=0.2, **kwargs):
        super().__init__(in_channels, block, cfg, num_classes)
        assert in_channels in (1,3)
        self.in_channels = in_channels
        self.flatten = nn.Flatten()

        self.topo_layer_11 = nn.Sequential(nn.Flatten(),    # remove this
                                        AdTopoLayer(out_features, m0_1, **kwargs),
                                        nn.ReLU())  # remove this
        self.topo_layer_21 = AdTopoLayer(out_features, m0_2, **kwargs)
        
        if in_channels == 3:
            self.topo_layer_12 = AdTopoLayer(out_features, m0_1, **kwargs)
            self.topo_layer_13 = AdTopoLayer(out_features, m0_1, **kwargs)
            self.topo_layer_22 = AdTopoLayer(out_features, m0_2, **kwargs)
            self.topo_layer_23 = AdTopoLayer(out_features, m0_2, **kwargs)
        
        # self.bn = nn.BatchNorm1d(2*in_channels*out_features)

        self.num_topo_layers = 0    # counts number of topo layers(3 if only m0=0.05 is used, 6 if m0=0.2 is also used)
        for m in self.modules():
            if isinstance(m, AdTopoLayer):
                self.num_topo_layers += 1
            elif isinstance(m, nn.BatchNorm2d): # initialize BN of topo_layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.relu = nn.ReLU()
        self.fc = nn.Linear((2**(5+self.num_res_layers))*block.expansion + self.num_topo_layers*out_features, num_classes)
    
    def forward(self, input):
        x = self.conv_layer(input)
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        # x = self.res_layer_4(x)
        x_0 = self.pool(x)

        if self.in_channels == 1:
            x_1 = self.topo_layer_11(self.flatten(input))
            if self.num_topo_layers == 2:
                x_2 = self.topo_layer_21(input)
                out = torch.concat((x_0, x_1, x_2), dim=-1)
            else:
                out = torch.concat((x_0, x_1), dim=-1)
        else:
            x_1 = self.topo_layer_11(self.flatten(input[:, [0], :, :]))
            x_2 = self.topo_layer_12(self.flatten(input[:, [1], :, :]))
            x_3 = self.topo_layer_13(self.flatten(input[:, [2], :, :]))
            if self.num_topo_layers == 6:
                x_4 = self.topo_layer_21(self.flatten(input[:, [0], :, :]))
                x_5 = self.topo_layer_22(self.flatten(input[:, [1], :, :]))
                x_6 = self.topo_layer_23(self.flatten(input[:, [2], :, :]))
                out = torch.concat((x_0, x_1, x_2, x_3, x_4, x_5, x_6), dim=-1)
            else:
                out = torch.concat((x_0, x_1, x_2, x_3), dim=-1)

        output = self.fc(out)
        return output

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


class ResNet18_8(ResNet):
    def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10, res_in_channels=8, p=0.1):     # 8/16/32/64
        super().__init__(in_channels, block, cfg, num_classes, res_in_channels, p)


class ResNet18_16(ResNet):
    def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10, res_in_channels=16, p=0.1):    # 16/32/64/128
        super().__init__(in_channels, block, cfg, num_classes, res_in_channels, p)


class ResNet18_32(ResNet):
    def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10, res_in_channels=32, p=0.1):    # 32/64/128/256
        super().__init__(in_channels, block, cfg, num_classes, res_in_channels, p)


# class ResNet34(ResNet):
#     def __init__(self, in_channels, block=ResidualBlock, cfg=[3,4,6,3], num_classes=7):
#         super().__init__(in_channels, block, cfg, num_classes)


# class AdPRNet18(AdPllayResNet):
#     def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], out_features=50, num_classes=7):
#         super().__init__(in_channels, block, cfg, out_features, num_classes)