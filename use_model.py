'''
Create ResNet models
'''
from model import SimpleBlock, BottleneckBlock, ResNet
from PIL import Image
from torchvision import transforms
from params import Params

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

def create_resnet(name: str, num_classes: int = Params.NUM_CLASSES, is_plain: bool = False) -> ResNet:
    if name not in resnet_models:
        raise ValueError(f'ERROR || {name} is not a valid ResNet model, please choose from: {list(resnet_models.keys())}')
    model_config = resnet_models[name]
    return ResNet(model_config['block'], model_config['num_blocks'], num_classes=num_classes, is_plain=is_plain)

def prepare_image(img_file: str) -> Image:
    """
    Prepare image for prediction
    """
    img = Image.open(img_file)
    '''
    PreProcess Image
    Based on section 3.4 on page 4 of the paper
        - Resize to 256x256
        - Center crop to 224x224
        - Normalize
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input = preprocess(img)
    # add batch dimension
    input = input.unsqueeze(0)
    return input