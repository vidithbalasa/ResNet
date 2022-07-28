'''
Create ResNet models
'''
from model import SimpleBlock, BottleneckBlock, ResNet
from image_process import prepare_image_pred

resnet_models = {
    # Simple Models
    'resnet18': {
        'block': SimpleBlock,
        'num_blocks': [2, 2, 2, 2],
    },
    'resnet34': {
        'block': SimpleBlock,
        'num_blocks': [3, 4, 6, 3],
    },
    # Bottleneck Models
    'resnet50': {
        'block': BottleneckBlock,
        'num_blocks': [3, 4, 6, 3],
    },
    'resnet101': {
        'block': BottleneckBlock,
        'num_blocks': [3, 4, 23, 3],
    },
    'resnet152': {
        'block': BottleneckBlock,
        'num_blocks': [3, 8, 36, 3],
    }
}

def create_resnet(name: str, num_classes: int = 10, is_plain: bool = False) -> ResNet:
    if name not in resnet_models:
        raise ValueError(f'ERROR || {name} is not a valid ResNet model, please choose from: {list(resnet_models.keys())}')
    model_config = resnet_models[name]
    return ResNet(model_config['block'], model_config['num_blocks'], num_classes=num_classes, is_plain=is_plain)

def get_test_image(file: str = 'ILSVRC2012_val_00000907.JPEG'):
    '''
    Loads a test image from the ILSVRC2012 dataset.
    '''
    return prepare_image_pred(file)