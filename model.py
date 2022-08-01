import torch
import torch.nn as nn
from params import Params
from typing import List, Optional

class ResNet(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: List[int], num_classes: int=Params.NUM_CLASSES, is_plain: bool=False):
        '''
        A convolutional neural network with residual connections.
            @param block: the block to use in the ResNet
            @param num_blocks: a list of the number of blocks in each layer
            @param num_classes: the number of classes in the dataset
            @param is_plain: doesn't use residual connections if True
        '''
        super(ResNet, self).__init__()
        self.is_plain = is_plain
        assert len(num_blocks) == 4, 'num_blocks must be a list of length 4, pad ending with 0 to use less layers'
        # num of input / output channels based on block type
        if block == SimpleBlock:
            # only using simple block for CIFAR data (3 layers instead of 4)
            channels = [(16, 16), (16, 32), (32, 64)]
            kernel, stride, pad = 3, 1, 1
            out_channels = 16
        elif block == BottleneckBlock:
            channels = [(64,256), (256, 512), (512, 1024), (1024, 2048)]
            kernel, stride, pad = 7, 2, 3
            out_channels = 64
        '''
        Follows the ResNet architecture shown on page 4 of the paper.
            - Open with a 7x7 kernel conv layer
            - Add a max pooling layer with a 2x2 kernel and stride of 2
            BLOCKS (SimpleBlock or BottleneckBlock)
                - Add blocks increasing in number of kernels from 64 to 512
                - input and output channels are based on block type (see `channels` above)
            - End with a 1000 neuron output layer (with a softmax activation)
        '''
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=pad)
        self.norm = nn.BatchNorm2d(num_features=out_channels, track_running_stats=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # BLOCKS
        self.conv2_x = self._make_layer(block, num_blocks[0], channels[0][0], channels[0][1])
        self.conv3_x = self._make_layer(block, num_blocks[1], channels[1][0], channels[1][1])
        self.conv4_x = self._make_layer(block, num_blocks[2], channels[2][0], channels[2][1])
        self.conv5_x = self._make_layer(block, num_blocks[3], channels[3][0], channels[3][1]) if num_blocks[3] else None
        # OUTPUT
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(in_features=channels[-1][-1], out_features=num_classes, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def _make_layer(self, block, num_blocks, in_channels, out_channels) -> Optional[nn.Sequential]:
        if num_blocks == 0: 
            return
        layers = []
        # first layer needs to handle change in # of kernels
        layers.append(block(in_channels, out_channels, is_plain=self.is_plain))
        # subsequent layers
        for _ in range(num_blocks):
            layers.append(block(out_channels, out_channels, is_plain=self.is_plain))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        if self.conv5_x is not None:
            x = self.conv5_x(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        x = self.softmax(x)

        return x

'''
Blocks

    SimpleBlock
        - 2 convolutional layers w a 3x3 kernel
        - Stride increases to 2 if # of input channels != # of output channels
    
    BottleneckBlock
        - 3 convolutional layers w a 1x1 kernel, a 3x3 kernel, and a 1x1 kernel
            - the first kernel downsamples the input (e.g. 64 -> 32)
            - the second kernel learns the features of the input
            - the third kernel upsamples the output (e.g. 32 -> 64)
        - Stride increases to 2 if # of input channels != # of output channels
'''
class SimpleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_plain: bool=False):
        super(SimpleBlock, self).__init__()
        self.is_plain = is_plain
        self.num_channels_is_same = in_channels == out_channels
        stride = 1 if self.num_channels_is_same else 2
        ''' Architecture '''
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=True)
        # layer 1 will spit out `out_channels` so this becomes the input to layer 2
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=True)
        self.relu = nn.ReLU()
        if not self.num_channels_is_same:
            # use a dense layer to project the identity to the output channels
            self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)
    
    # shortcut connection described by option A in the paper
    def shortcut_connection(self, identity: torch.Tensor) -> torch.Tensor:
        '''
        Shortcut connection described by option A in the paper
            1. downsample identity to match output (because first block downsamples w stride of 2)
            2. zero pad identity to match output channels (double identity channels)
        '''
        identity = self.downsample(identity)
        # double number of identity channels
        # first turn identity into a zero valued tensor
        identity_zero_clone = torch.zeros_like(identity)
        # then combine identity with the identity zero clone
        identity = torch.cat([identity, identity_zero_clone], dim=1)
        return identity
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.bn2(x)

        if self.is_plain: return x

        ''' Shortcut Connection '''
        if not self.num_channels_is_same:
            # project the input to the output channels
            identity = self.shortcut_connection(identity)
        x += identity
        x = self.relu(x)
        
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_plain: bool=False):
        super(BottleneckBlock, self).__init__()
        self.is_plain = is_plain
        # store if the num of channels changes
        self.num_channels_is_same = in_channels == out_channels
        if not self.num_channels_is_same:
            self.projection_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        # downsample the input by 2 for the first conv layer and 4 for the rest
        downsampled_channels = in_channels // 4 if self.num_channels_is_same else in_channels // 2
        '''
        On page 5 of the paper, conv2_1 is the only block that doesn't downsample the original input.
        Instead it keeps the channel size at 64 and just upsamples to 256. This is because conv1 down-
        samples the input to 64 channels, while conv2_x upsamples to 256. The rest of the blocks
        downsample the first input by half then upsample. So, if the input is 64 then we shouldn't 
        downsample it.
        '''
        if in_channels != out_channels and in_channels == 64:
            downsampled_channels = in_channels
        # increase stride if num input channels != num output channels
        stride = 1 if in_channels == out_channels else 2
        '''
        1x1 conv - downsample input
        3x3 conv - learn features
        1x1 conv - upsample output
        '''
        self.conv1 = nn.Conv2d(in_channels, downsampled_channels, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(downsampled_channels, track_running_stats=True)
        self.conv2 = nn.Conv2d(downsampled_channels, downsampled_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(downsampled_channels, track_running_stats=True)
        self.conv3 = nn.Conv2d(downsampled_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels, track_running_stats=True)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.is_plain: return x

        '''
        Residual Conection
        This acts as the residual layer where the previous output is added
        to the current output. This allows the model to change the identity instead
        of changing a 0-biased function.
        '''
        if not self.num_channels_is_same:
            identity = self.projection_layer(identity)
        x += identity
        x = self.relu(x)

        return x