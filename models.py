import torch
import torch.nn as nn
import numpy as np
from dtm import DTMLayer
from eclay import EC_TopoLayer
from pllay import PL_TopoLayer


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
    def __init__(self, in_channels, block=ResidualBlock, block_cfg=[2,2,2,2], filter_cfg=[16,32,64,128], num_classes=10):
        """
        Args:
            in_channels: Number of channels of input data
            block: Type of block to use. ResidualBlock or BottleneckBlock
            block_cfg: List containing number of blocks for each layer
            channel_cfg: List containing number of filters at the start of each layer
            num_classes:
        """
        super().__init__()
        assert len(block_cfg) == len(filter_cfg)
        self.layer_input_channels = filter_cfg[0]   # channel of input that goes into res_layer1, value changes in _make_layers

        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, filter_cfg[0], kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.layer_input_channels),
                                        nn.ReLU())
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # self.res_layer_1 = self._make_layers(block, res_in_channels, cfg[0], stride=1)
        # self.res_layer_2 = self._make_layers(block, 2*res_in_channels, cfg[1], stride=2)
        # self.res_layer_3 = self._make_layers(block, 4*res_in_channels, cfg[2], stride=2)
        # self.res_layer_4 = self._make_layers(block, 8*res_in_channels, cfg[3], stride=2)

        self.res_layers = nn.Sequential(*[self._make_layers(block, num_filters, num_blocks, stride=(1 if i==0 else 2)) for i, (num_blocks, num_filters) in enumerate(zip(block_cfg, filter_cfg))])

        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
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
        output = self.fc(x)
        return output


class EClayResNet(ResNet):
    def __init__(self, in_channels, block=ResidualBlock, block_cfg=[2,2], filter_cfg=[64, 128], num_classes=10,     # resnet params
                 load_res=False, res_path="./MNIST/saved_weights/ResNet_MNIST/00_00/sim1.pt", freeze_res=True,      # loading pretrained resnet
                 start=0, end=7, T=32, num_channels=1, hidden_features=[64, 32],                                    # EC params
                 start_2=1, end_2=8,                                                                                # EC params 2
                 load_ec=False, ec_path="./MNIST/saved_weights/EClay_MNIST/00_00/sim1.pt", freeze_ec=True,          # loading pretrained eclay
                 use_dtm=True, m0_1=0.05, m0_2=0.2, **kwargs): # dtm params
        super().__init__(in_channels, block, block_cfg, filter_cfg, num_classes)
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = EC_TopoLayer(False, start, end, T, num_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = EC_TopoLayer(False, start_2, end_2, T, num_channels, hidden_features)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(filter_cfg[-1] + 2*hidden_features[-1], num_classes)

        # self.num_topo_layers = 0    # counts number of topo layers(3 if only m0=0.05 is used, 6 if m0=0.2 is also used)
        # for m in self.modules():
        #     if isinstance(m, AdTopoLayer):
        #         self.num_topo_layers += 1
        #     elif isinstance(m, nn.BatchNorm2d): # initialize BN of topo_layers
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        if load_res:
            self._load_pretrained_resnet(res_path, freeze=freeze_res)
        if load_ec:
            self._load_pretrained_eclay(ec_path, freeze=freeze_ec)
    
    def forward(self, input):
        # ResNet
        x_1 = self.conv_layer(input)
        # x = self.max_pool(x)
        x_1 = self.res_layers(x_1)
        x_1 = self.avg_pool(x_1)

        # EC Layer 1
        x_2 = self.dtm_1(input)
        x_2 = self.topo_layer_1(x_2)
        x_2 = self.relu(x_2)

        # EC Layer 2
        x_3 = self.dtm_2(input)
        x_3 = self.topo_layer_2(x_3)
        x_3 = self.relu(x_3)

        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x

    def _load_pretrained_eclay(self, weight_path_1,
                              weigth_path_2=None, freeze=False):
        model_dict = self.state_dict()
        
        eclay_pretrained_1 = torch.load(weight_path_1)
        eclay_params_1 = {layer:params for layer, params in eclay_pretrained_1.items() if layer in model_dict and "fc" not in layer}
        model_dict.update(eclay_params_1)

        # if not (weigth_path_2 is None):
        #     pllay_pretrained_2 = torch.load(weigth_path_2)
        #     pllay_params_2 = {layer:params for layer, params in pllay_pretrained_2.items() if layer in model_dict and "fc" not in layer}
        #     model_dict.update(pllay_params_2)
        
        self.load_state_dict(model_dict)

        if freeze:
            for m in self.modules():
                if isinstance(m, EC_TopoLayer):
                    m.requires_grad_(False)

    def _load_pretrained_resnet(self, weight_path, freeze=False):
        model_dict = self.state_dict()

        resnet_pretrained = torch.load(weight_path)
        resnet_params = {layer:params for layer, params in resnet_pretrained.items() if layer in model_dict and "fc" not in layer}
        model_dict.update(resnet_params)

        self.load_state_dict(model_dict)

        if freeze:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
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


class CNN_2(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1))
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(nn.Linear(784, 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, input):
        x = self.conv_layer(input)
        x = self.flatten(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class EC_CNN_2(CNN_2):
    def __init__(self, in_channels=1, num_classes=10,   # CNN params
                 start=0, end=7, T=32, num_channels=1, hidden_features=[64, 32],   # EC parameters
                 start_2=1, end_2=8,                            # EC parameters 2
                 use_dtm=True, m0_1=0.05, m0_2=0.2, **kwargs):  # DTM parameters
        super().__init__(in_channels, num_classes)
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = EC_TopoLayer(False, start, end, T, num_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = EC_TopoLayer(False, start_2, end_2, T, num_channels, hidden_features)
        self.fc = nn.Sequential(nn.Linear(784 + 2*hidden_features[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))


    def forward(self, input):
        # CNN
        x_1 = self.conv_layer(input)
        x_1 = self.flatten(x_1)
        
        # EC Layer 1
        x_2 = self.dtm_1(input)
        x_2 = self.topo_layer_1(x_2)

        # EC Layer 2
        x_3 = self.dtm_2(input)
        x_3 = self.topo_layer_2(x_3)

        # FC Layer
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.relu(x)
        x = self.fc(x)
        return x


class PL_CNN_2(CNN_2):
    def __init__(self, in_channels=1, num_classes=10,   # CNN params
                 start=0, end=7, T=32, K_max=2, dimensions=[0, 1], num_channels=1, hidden_features=[32],   # PL parameters
                 start_2=1, end_2=8, K_max_2=3,                 # PL parameters 2
                 use_dtm=True, m0_1=0.05, m0_2=0.2, **kwargs):  # DTM parameters
        super().__init__(in_channels, num_classes)
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = PL_TopoLayer(False, start, end, T, K_max, dimensions, num_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = PL_TopoLayer(False, start_2, end_2, T, K_max_2, dimensions, num_channels, hidden_features)
        self.fc = nn.Sequential(nn.Linear(784 + 2*hidden_features[-1], 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, input):
        # CNN
        x_1 = self.conv_layer(input)
        x_1 = self.flatten(x_1)
        
        # PL Layer 1
        x_2 = self.dtm_1(input)
        x_2 = self.topo_layer_1(x_2)

        # PL Layer 2
        x_3 = self.dtm_2(input)
        x_3 = self.topo_layer_2(x_3)

        # FC Layer
        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.relu(x)
        x = self.fc(x)
        return x