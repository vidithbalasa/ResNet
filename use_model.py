'''
Create ResNet models
'''
from model import SimpleBlock, BottleneckBlock, ResNet
from params import Params

def create_resnet(architecture: dict, num_classes: int = Params.NUM_CLASSES, is_plain: bool = False) -> ResNet:
    assert 'block' in architecture and 'num_blocks' in architecture, 'param `architecture` must have a block and num_blocks key'
    return ResNet(architecture['block'], architecture['num_blocks'], num_classes=num_classes, is_plain=is_plain)

class Architectures:
    # Simple Models
    resnet20 = {
        'block': SimpleBlock,
        'num_blocks': [3, 3, 3, 0]
    }
    resnet32 = {
        'block': SimpleBlock,
        'num_blocks': [5, 5, 5, 0]
    }
    resnet44 = {
        'block': SimpleBlock,
        'num_blocks': [7, 7, 7, 0]
    }
    resnet56 = {
        'block': SimpleBlock,
        'num_blocks': [9, 9, 9, 0]
    }
    # Bottleneck Models
    resnet50 = {
        'block': BottleneckBlock,
        'num_blocks': [3, 4, 6, 3]
    }
    resnet101 = {
        'block': BottleneckBlock,
        'num_blocks': [3, 4, 23, 3]
    }
    resnet152 = {
        'block': BottleneckBlock,
        'num_blocks': [3, 8, 36, 3]
    }