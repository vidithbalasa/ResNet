import torch
import torch.nn as nn

class ResNet(nn.Module):
    '''
    A ResNet model built on Blocks of Convolutional and Residual Layers.
        It takes in 
    '''
    def __init__(self, block: nn.Module, num_blocks: list[int]):
        super(ResNet, self).__init__()
        assert len(num_blocks) == 4, 'num_blocks must be a list of length 4'
        '''
        Follow the ResNet architecture shown on page 4 of the paper.
            - Open with a 7x7 kernel
            - Add a max pooling layer with a 2x2 kernel and stride of 2
            BLOCKS (SimpleBlock or BottleneckBlock)
                - Add blocks increasing in number of kernels from 64 to 512
                - Add a residual connection between each block
                    - Add a projection shortcut if the number of kernels is different between the two blocks
                    - Add a stride of 2 if changing # of kernels
            - End with a 1000 neuron output layer (with a softmax activation)
        '''
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.norm = nn.BatchNorm2d(num_features=64)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # BLOCKS
        self.block_layer1 = self._make_layer(block, num_blocks[0], 64, 64)
        self.block_layer2 = self._make_layer(block, num_blocks[1], 64, 128)
        self.block_layer3 = self._make_layer(block, num_blocks[2], 128, 256)
        self.block_layer4 = self._make_layer(block, num_blocks[3], 256, 512)
        # OUTPUT LAYER (a dense 1000 perceptron layer with a softmax activation)
        self.output_layer = nn.Linear(in_features=512, out_features=1000)
        self.softmax = nn.Softmax(dim=1)
    
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        # first layer needs to handle change in # of kernels
        layers.append(block(in_channels, out_channels))
        for _ in range(num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.norm(x)
        x = self.activation(x)

        x = self.maxpool(x)

        x = self.block_layer1(x)
        x = self.block_layer2(x)
        x = self.block_layer3(x)
        x = self.block_layer4(x)

        x = self.output_layer(x)
        x = self.softmax(x)

        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

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
    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleBlock, self).__init__()
        # handle case where # of channels changes
        if in_channels != out_channels:
            self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            # create 2 conv layers
            self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # norm layer
        self.norm = nn.BatchNorm2d(out_channels)
        # relu layer (not optimal but same as paper)
        self.activation = nn.ReLU(inplace=True)
        # layer 1 will spit out `out_channels` so this becomes the input to layer 2
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # run the first layer
        x = self.layer1(x)
        # add batch norm and ReLU
        x = self.norm(x)
        x = self.activation(x)
        # run the second layer
        x = self.layer2(x)
        # add batch norm and ReLU
        x = self.norm(x)
        x = self.activation(x)

        x += identity
        x = self.activation(x)
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(BottleneckBlock, self).__init__()
        # downsample the input
        downsampled_channels = in_channels // 4
        self.layer1 = nn.Conv2d(in_channels, downsampled_channels, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(downsampled_channels, downsampled_channels, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(downsampled_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # normalization & activation
        self.norm1 = nn.BatchNorm2d(downsampled_channels) # for the first 2 layers
        self.norm2 = nn.BatchNorm2d(out_channels) # last layer
        self.activation = nn.ReLU(inplace=True)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.layer1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.layer3(x)
        x = self.norm2(x)
        x = self.activation(x)

        '''
        Residual Conection
        This acts as the residual layer where the previous output is added
        to the current output. This allows the model to change the identity instead
        of changing a 0-biased function.
        '''
        x += identity
        x = self.activation(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)