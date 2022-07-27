import torch
import torch.nn as nn

class ResNet(nn.Module):
    '''
    A ResNet model built on Blocks of Convolutional Layers and Residual Layers.
        It takes in 
    '''
    def __init__(self):
        super(ResNet, self).__init__()
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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # TODO: Add blocks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
