from PIL import Image
from torchvision import transforms

def prepare_image_pred(img_file: str) -> Image:
    """
    Prepare image for processing
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

def prepare_image_train(img_file: str) -> Image:
    '''
    Prepare image for training (with augmentations)
    '''
    img = Image.open(img_file)
    '''
    PreProcess Image
    Based on section 3.4 on page 4 of the paper
        - Resize to 256x256
        - Randomly crop to 224x224
        - Randomly flip horizontally
        - Color augmentation
        - Normalize
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input = preprocess(img)
    return input
