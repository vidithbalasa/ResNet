from main import create_resnet
import torch

def test_resnet():
    checkpoint = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    model = create_resnet('resnet50', num_classes=1000)

    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
    return model