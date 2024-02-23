import torch
import torch.nn as nn
import numpy as np
from pllay_adap import AdTopoLayer
from dtm import DTMLayer
from ec import EC_TopoLayer


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
        output = self.relu(x)
        return output


class BottleneckBlock(nn.Module):
    expansion = 3
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(out_channels))
        self.conv_layer2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                        nn.BatchNorm2d(out_channels))
        self.conv_layer3 = nn.Sequential(nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(out_channels * self.expansion))
        self.downsample = downsample
        self.relu = nn.ReLU() 

    def forward(self, input):
        x = self.conv_layer1(input)
        x = self.relu(x)
        x = self.conv_layer2(x)
        x = self.relu(x)
        x = self.conv_layer3(x)
        x = x + input if self.downsample is None else x + self.downsample(input)
        output = self.relu(x)
        return output


class ResNet(nn.Module):
    def __init__(self, in_channels, block=ResidualBlock, block_cfg=[2,2,2,2], filter_cfg=[16,32,64,128], num_classes=10, p=0.1):
        """
        Args:
            in_channels: Number of channels of input data
            block: Type of block to use. ResidualBlock or BottleneckBlock
            block_cfg: List containing number of blocks for each layer
            channel_cfg: List containing number of filters at the start of each layer
            num_classes:
            p: Dropout percentage of fc layer
        """
        super().__init__()
        assert len(block_cfg) == len(filter_cfg)
        self.layer_input_channels = filter_cfg[0]   # channel of input that goes into res_layer1, value changes in _make_layers


        # original ResNet
        # self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, self.res_in_channels, kernel_size=7, stride=2, padding=3),
        #                                 nn.BatchNorm2d(self.res_in_channels),
        #                                 nn.ReLU())
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # modified ResNet
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, filter_cfg[0], kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.layer_input_channels),
                                        nn.ReLU())
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # self.res_layer_1 = self._make_layers(block, res_in_channels, cfg[0], stride=1)
        # self.res_layer_2 = self._make_layers(block, 2*res_in_channels, cfg[1], stride=2)
        # self.res_layer_3 = self._make_layers(block, 4*res_in_channels, cfg[2], stride=2)
        # self.res_layer_4 = self._make_layers(block, 8*res_in_channels, cfg[3], stride=2)

        # self.res_layers = nn.Sequential(*[self._make_layers(block, res_in_channels * (2**i), num_blocks, stride=(1 if i==0 else 2)) for i, num_blocks in enumerate(block_cfg)])
        self.res_layers = nn.Sequential(*[self._make_layers(block, num_filters, num_blocks, stride=(1 if i==0 else 2)) for i, (num_blocks, num_filters) in enumerate(zip(block_cfg, filter_cfg))])

        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(filter_cfg[-1], num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, num_filters, num_blocks, stride):
        """
        Args:
            block: Type of block to use. ResidualBlock or BottleneckBlock
            num_filters: Number of filters at the start of each layer
            num_blocks: Number of blocks for each layer
            stride:
        """  
        if stride != 1 or self.layer_input_channels != num_filters * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.layer_input_channels, num_filters*block.expansion, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(num_filters, block.expansion))
        else:
            downsample = None
        
        block_list =[]
        block_list.append(block(self.layer_input_channels, num_filters, stride, downsample))

        self.layer_input_channels = num_filters * block.expansion
        
        for _ in range(1, num_blocks):
            block_list.append(block(self.layer_input_channels, num_filters))
        return nn.Sequential(*block_list)

    def forward(self, input):
        x = self.conv_layer(input)
        # x = self.max_pool(x)
        x = self.res_layers(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        output = self.fc(x)
        return output


class EClayResNet(ResNet):
    def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10, res_in_channels=8, p_res=0,    # resnet params
                 T=200, num_channels=1, out_features=64, p_topo=0., # EC_Topolayer params
                 use_dtm=True, **kwargs): # dtm params
        super().__init__(in_channels, block, cfg, num_classes, res_in_channels, p_res)
        self.use_dtm = use_dtm
        if use_dtm:
            self.dtm = DTMLayer(**kwargs, scale_dtm=True)
        self.topo_layer = EC_TopoLayer(T, num_channels, out_features, p_topo)
        self.relu = nn.ReLU()

        self.linear = nn.Linear(2*out_features, num_classes)
        
        # self.bn = nn.BatchNorm1d(2*in_channels*out_features)

        # self.num_topo_layers = 0    # counts number of topo layers(3 if only m0=0.05 is used, 6 if m0=0.2 is also used)
        # for m in self.modules():
        #     if isinstance(m, AdTopoLayer):
        #         self.num_topo_layers += 1
        #     elif isinstance(m, nn.BatchNorm2d): # initialize BN of topo_layers
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        self.load_pretrained_pllay()
    
    def forward(self, input):
        # ResNet
        x = self.conv_layer(input)
        # x = self.max_pool(x)
        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x = self.avg_pool(x)
        # x = self.dropout(x)

        # EClay
        if self.use_dtm:
            x_1 = self.dtm(input)
        else:
            x_1 = input
        x_1 = self.relu(self.topo_layer(x_1))
        # x = self.bn(x)  ################################## whether to use this or not

        out = torch.concat((x, x_1), dim=-1)
        output = self.linear(self.dropout(out))
        return output

    def load_pretrained_pllay(self, pllay_pretrained_file_1="./MNIST/saved_weights/EClay64_MNIST_500/00_00/sim1.pt",
                              pllay_pretrained_file_2=None, freeze=True):
        model_dict = self.state_dict()
        
        pllay_pretrained_1 = torch.load(pllay_pretrained_file_1)
        pllay_params_1 = {layer:params for layer, params in pllay_pretrained_1.items() if layer in model_dict and "fc" not in layer}
        print(pllay_params_1)
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


# class ResNet18_8(ResNet):
#     def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10, res_in_channels=8, p=0.1):     # 8/16/32/64
#         super().__init__(in_channels, block, cfg, num_classes, res_in_channels, p)


# class ResNet18_16(ResNet):
#     def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10, res_in_channels=16, p=0.1):    # 16/32/64/128
#         super().__init__(in_channels, block, cfg, num_classes, res_in_channels, p)


# class ResNet18_32(ResNet):
#     def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], num_classes=10, res_in_channels=32, p=0.1):    # 32/64/128/256
#         super().__init__(in_channels, block, cfg, num_classes, res_in_channels, p)


# class ResNet34(ResNet):
#     def __init__(self, in_channels, block=ResidualBlock, cfg=[3,4,6,3], num_classes=7):
#         super().__init__(in_channels, block, cfg, num_classes)


# class AdPRNet18(AdPllayResNet):
#     def __init__(self, in_channels, block=ResidualBlock, cfg=[2,2,2,2], out_features=50, num_classes=7):
#         super().__init__(in_channels, block, cfg, out_features, num_classes)