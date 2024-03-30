import torch
import torch.nn as nn
from dtm import DTMLayer
from eclay import ECLay
from pllay import PLLay


# PLNet
class PLNet(nn.Module):
    def __init__(self, num_classes,
                 tseq=[0, 7, 32], K_max=2, dimensions=[0, 1], in_channels=1, hidden_features=[32],  # PL params
                 use_dtm=True, **kwargs):                                                           # DTM params
        """_summary_

        Args:
            num_classes (_type_): _description_
            start (int, optional): _description_. Defaults to 0.
            end (int, optional): _description_. Defaults to 7.
            T (int, optional): _description_. Defaults to 32.
            K_max (int, optional): _description_. Defaults to 2.
            dimensions (list, optional): _description_. Defaults to [0, 1].
            in_channels (int, optional): _description_. Defaults to 1.
            hidden_features (list, optional): _description_. Defaults to [32].
        """
        super().__init__()
        self.use_dtm = use_dtm
        if use_dtm:
            self.dtm = DTMLayer(**kwargs)

        superlevel = False if use_dtm else True
        self.topo_layer = PLLay(superlevel, tseq, K_max, dimensions, in_channels, hidden_features)
        self.fc = nn.Linear(hidden_features[-1], num_classes)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]

        Returns:
            output: Tensor of shape [batch_size, num_classes]
        """
        if self.use_dtm:
            x = self.dtm(input)
        else:
            x = input
        x = self.topo_layer(x)
        output = self.fc(x)
        return output


class PLNet2(nn.Module):
    def __init__(self, num_classes,
                 tseq_1=[0, 7, 32], K_max_1=2, dimensions=[0, 1], in_channels=1, hidden_features=[32],  # PL params
                 tseq_2=[1, 8, 32], K_max_2=3,                                                          # PL params 2
                 m0_1=0.05, m0_2=0.2, **kwargs):                                                        # DTM params
        """
        Args:
            out_features: output dimension of fc layer
            num_classes: number of classes for classification
            use_dtm: whether to use DTM filtration
            kwargs: parameters for dtm
                    ex) m0=0.05, lims=[[1,28], [1,28]], size=[28, 28], r=2
        """
        super().__init__()
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = PLLay(False, tseq_1, K_max_1, dimensions, in_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = PLLay(False, tseq_2, K_max_2, dimensions, in_channels, hidden_features)
        self.fc = nn.Linear(2*hidden_features[-1], num_classes)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]

        Returns:
            output: Tensor of shape [batch_size, num_classes]
        """
        x_1 = self.dtm_1(input)
        x_1 = self.topo_layer_1(x_1)

        x_2 = self.dtm_2(input)
        x_2 = self.topo_layer_2(x_2)

        x = torch.concat((x_1, x_2), dim=-1)
        output = self.fc(x)
        return output


# ECNet
class ECNet(nn.Module):
    def __init__(self, num_classes,
                 tseq=[0, 7, 32], in_channels=1, hidden_features=[64, 32],  # EC params
                 use_dtm=True, **kwargs):                                   # DTM params
        """
        Args:
            out_features: output dimension of fc layer
            num_classes: number of classes for classification
            use_dtm: whether to use DTM filtration
            kwargs: parameters for dtm
                    ex) m0=0.05, lims=[[1,28], [1,28]], size=[28, 28], r=2
        """
        super().__init__()
        self.use_dtm = use_dtm
        if use_dtm:
            self.dtm = DTMLayer(**kwargs)

        superlevel = False if use_dtm else True
        self.topo_layer = ECLay(superlevel, tseq, in_channels, hidden_features)
        self.fc = nn.Linear(hidden_features[-1], num_classes)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]

        Returns:
            output: Tensor of shape [batch_size, num_classes]
        """
        if self.use_dtm:
            x = self.dtm(input)
        else:
            x = input
        x = self.topo_layer(x)
        output = self.fc(x)
        return output


class ECNet2(nn.Module):
    def __init__(self, num_classes,
                 tseq_1=[0, 7, 32], in_channels=1, hidden_features=[64, 32],    # EC params
                 tseq_2=[1, 8, 32],                                             # EC params 2
                 m0_1=0.05, m0_2=0.2, **kwargs):                                # DTM params
        """
        Args:
            out_features: output dimension of fc layer
            num_classes: number of classes for classification
            use_dtm: whether to use DTM filtration
            kwargs: parameters for dtm
                    ex) m0=0.05, lims=[[1,28], [1,28]], size=[28, 28], r=2
        """
        super().__init__()
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = ECLay(False, tseq_1, in_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = ECLay(False, tseq_2, in_channels, hidden_features)
        self.fc = nn.Linear(2*hidden_features[-1], num_classes)

    def forward(self, input):
        """
        Args:
            input: Tensor of shape [batch_size, num_channels, H, W]

        Returns:
            output: Tensor of shape [batch_size, num_classes]
        """
        x_1 = self.dtm_1(input)
        x_1 = self.topo_layer_1(x_1)

        x_2 = self.dtm_2(input)
        x_2 = self.topo_layer_2(x_2)

        x = torch.concat((x_1, x_2), dim=-1)
        output = self.fc(x)
        return output


# CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(784, 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, input):
        x = self.conv_layer(input)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ECCNN(CNN):
    def __init__(self, in_channels=1, num_classes=10,           # CNN params
                 tseq_1=[0, 7, 32], hidden_features=[64, 32],   # EC params
                 tseq_2=[1, 8, 32],                             # EC params 2
                 m0_1=0.05, m0_2=0.2, **kwargs):                # DTM params
        super().__init__(in_channels, num_classes)
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = ECLay(False, tseq_1, in_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = ECLay(False, tseq_2, in_channels, hidden_features)
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
        x = self.fc(x)
        return x


class PLCNN(CNN):
    def __init__(self, in_channels=1, num_classes=10,                                   # CNN params
                 tseq_1=[0, 7, 32], K_max_1=2, dimensions=[0, 1], hidden_features=[32], # PL params
                 tseq_2=[1, 8, 32], K_max_2=3,                                          # PL params 2
                 m0_1=0.05, m0_2=0.2, **kwargs):                                        # DTM params
        super().__init__(in_channels, num_classes)
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = PLLay(False, tseq_1, K_max_1, dimensions, in_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = PLLay(False, tseq_2, K_max_2, dimensions, in_channels, hidden_features)
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
        x = self.fc(x)
        return x


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
        output = self.relu(x)
        return output


class ResNet(nn.Module):
    def __init__(self, in_channels, block=ResidualBlock, block_cfg=[2, 2], filter_cfg=[64, 128], num_classes=10):
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
        self.layer_input_channels = in_channels # channel of input that goes into res_layer1, value changes in _make_layers
        
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

    def forward(self, x):
        x = self.res_layers(x)
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


class ECResNet(ResNet):
    def __init__(self, in_channels, block=ResidualBlock, block_cfg=[2,2], filter_cfg=[64, 128], num_classes=10, # ResNet params
                 tseq_1=[0, 7, 32], hidden_features=[64, 32],                                                   # EC params
                 tseq_2=[1, 8, 32],                                                                             # EC params 2
                 m0_1=0.05, m0_2=0.2, **kwargs):                                                                # DTM params
        super().__init__(in_channels, block, block_cfg, filter_cfg, num_classes)
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = ECLay(False, tseq_1, in_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = ECLay(False, tseq_2, in_channels, hidden_features)
        self.fc = nn.Linear(filter_cfg[-1] + 2*hidden_features[-1], num_classes)
    
    def forward(self, input):
        # ResNet
        x_1 = self.res_layers(input)
        x_1 = self.avg_pool(x_1)

        # EC Layer 1
        x_2 = self.dtm_1(input)
        x_2 = self.topo_layer_1(x_2)

        # EC Layer 2
        x_3 = self.dtm_2(input)
        x_3 = self.topo_layer_2(x_3)

        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x


class PLResNet(ResNet):
    def __init__(self, in_channels, block=ResidualBlock, block_cfg=[2,2], filter_cfg=[64, 128], num_classes=10, # ResNet params
                 tseq_1=[0, 7, 32], K_max_1=2, dimensions=[0, 1], hidden_features=[32],                         # PL params
                 tseq_2=[1, 8, 32], K_max_2=3,                                                                  # PL params 2
                 m0_1=0.05, m0_2=0.2, **kwargs):                                                                # DTM params
        super().__init__(in_channels, block, block_cfg, filter_cfg, num_classes)
        self.dtm_1 = DTMLayer(m0=m0_1, **kwargs)
        self.topo_layer_1 = PLLay(False, tseq_1, K_max_1, dimensions, in_channels, hidden_features)
        self.dtm_2 = DTMLayer(m0=m0_2, **kwargs)
        self.topo_layer_2 = PLLay(False, tseq_2, K_max_2, dimensions, in_channels, hidden_features)
        self.fc = nn.Linear(filter_cfg[-1] + 2*hidden_features[-1], num_classes)
    
    def forward(self, input):
        # ResNet
        x_1 = self.res_layers(input)
        x_1 = self.avg_pool(x_1)

        # PL Layer 1
        x_2 = self.dtm_1(input)
        x_2 = self.topo_layer_1(x_2)

        # PL Layer 2
        x_3 = self.dtm_2(input)
        x_3 = self.topo_layer_2(x_3)

        x = torch.concat((x_1, x_2, x_3), dim=-1)
        x = self.fc(x)
        return x

