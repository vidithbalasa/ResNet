import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from params import Params
from typing import Optional
from tqdm import tqdm
from datetime import datetime
from data_loader import get_CIFAR_data

def train(model: nn.Module, epochs: int=1, save_name: str='') -> None:
    '''
    Train the model.
    '''
    trainloader, testloader, classes = get_CIFAR_data()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Params.LEARNING_RATE)
    if save_name:
        with open(f'{save_name}.csv', 'w') as f:
            f.write(f'epoch,train_loss,train_accuracy,test_loss,test_accuracy\n')
    
    for epoch_idx in range(epochs):
        model.train(True)
        train_loss, train_accuracy = train_one_epoch(model, trainloader, optimizer, loss_fn, epoch_idx)
        model.eval()
        test_loss, test_accuracy = evaluate_model(model, testloader, loss_fn)
        print(f'\tTrain Loss: {train_loss:.2f} - Test Loss: {test_loss:.2f} - Test Accuracy: {test_accuracy*100:.2f}%')
        if save_name:
            with open(f'{save_name}.csv', 'a') as f:
                f.write(f'{epoch_idx},{train_loss:.2f},{train_accuracy:.2f}%,{test_loss:.2f},{test_accuracy*100:.2f}%\n')

    # save model
    torch.save(model.state_dict(), f'{save_name}.pt')
    print(f'Model saved to {save_name}.pt')


def train_one_epoch(
    model: nn.Module, 
    trainloader: DataLoader, 
    optimizer: torch.optim, 
    loss_fn: nn.modules.loss, 
    epoch_idx: int,
) -> float and float:
        '''
        Train a single epoch.
        '''
        running_loss = 0
        correct = 0

        # tqdm loader with title 'epoch {epoch_idx}'
        for batch_idx, (images, labels) in tqdm(enumerate(trainloader), desc=f'Epoch {epoch_idx}'):
            images, labels = images.to(Params.DEVICE), labels.to(Params.DEVICE)
            # zero gradients for each batch
            optimizer.zero_grad()
            # get model predictions
            outputs = model(images)
            # check if the model is correct
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # compute loss & gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
            # backpropagate
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(trainloader), correct / len(trainloader)

def evaluate_model(model: nn.Module, validloader: DataLoader, loss_fn: nn.modules.loss) -> float:
    '''
    Compute the accuracy and loss of the model on the validation set.
    '''
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images, labels = images.to(Params.DEVICE), labels.to(Params.DEVICE)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # compute loss
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(validloader), correct / total