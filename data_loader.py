import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from params import Params

def get_CIFAR_data() -> DataLoader and DataLoader and tuple:
    '''
    Get the data. 
    Dataset: CIFAR10
    '''
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, 
        download=True, transform=transformation
    )

    trainloader = DataLoader(
        trainset, batch_size=Params.BATCH_SIZE,
        shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transformation
    )
    testloader = DataLoader(
        testset, batch_size=Params.BATCH_SIZE,
        shuffle=False, num_workers=2
    )

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes