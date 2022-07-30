import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from params import Params
from typing import List, Any
from tqdm import tqdm
from datetime import datetime

def train(model: nn.Module, epochs: int=1) -> None:
    '''
    Train the model.
    '''
    trainloader, validloader, testloader, classes = get_CIFAR_data()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Params.LEARNING_RATE)
    
    for epoch in range(epochs):
        model.train(True)
        avg_loss = train_one_epoch(trainloader, model, optimizer, loss_fn, epoch+1)
        model.eval()
        accuracy = model_accuracy(model, validloader)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.2f} - Accuracy: {accuracy*100:.2f}%')

    # save model
    torch.save(model.state_dict(), f'model-{datetime.now().strftime("%Y%m%d-%H%M")}.pt')
    
    # get test accuracy
    model.eval()
    accuracy = model_accuracy(model, testloader)
    print(f'Test accuracy: {accuracy}')


def train_one_epoch(trainloader: DataLoader, model: nn.Module, optimizer: torch.optim, loss_fn: nn.modules.loss, epoch_idx: int) -> float:
        running_loss = 0

        # tqdm loader with title 'epoch {epoch_idx}'
        for batch_idx, (images, labels) in tqdm(enumerate(trainloader), desc=f'Epoch {epoch_idx}'):
            images, labels = images.to(Params.DEVICE), labels.to(Params.DEVICE)
            # zero gradients for each batch
            optimizer.zero_grad()
            # get model predictions
            outputs = model(images)
            # compute loss & gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
            # backpropagate
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(trainloader)

def model_accuracy(model: nn.Module, validloader: DataLoader) -> float:
    '''
    Compute the accuracy of the model on the validation set.
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images, labels = images.to(Params.DEVICE), labels.to(Params.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def get_CIFAR_data() -> DataLoader and DataLoader and DataLoader and tuple:
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

    # create validation set
    trainset, validset = torch.utils.data.random_split(trainset, [45000, 5000])

    trainloader = DataLoader(
        trainset, batch_size=Params.BATCH_SIZE,
        shuffle=True, num_workers=2
    )

    validloader = DataLoader(
        validset, batch_size=Params.BATCH_SIZE,
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

    return trainloader, validloader, testloader, classes