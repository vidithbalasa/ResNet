import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from params import Params
from typing import Optional
from tqdm import tqdm
from datetime import datetime
from data_loader import get_CIFAR_data

def train(model: nn.Module, epochs: int=1, save_results: bool=False) -> None:
    '''
    Train the model.
    '''
    trainloader, validloader, testloader, classes = get_CIFAR_data()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Params.LEARNING_RATE, momentum=Params.MOMENTUM)
    curr_time = datetime.now().strftime("%Y%M%D-%H%M")
    res_file = f'./results-{curr_time}.csv' if save_results else None
    if save_results:
        with open(res_file, 'w') as f:
            f.write(f'epoch,train_loss,valid_loss,valid_accuracy\n')
    
    for epoch_idx in range(1, epochs+1):
        model.train(True)
        train_loss = train_one_epoch(trainloader, model, optimizer, loss_fn, epoch_idx)
        model.eval()
        val_accuracy, val_loss = evaluate_model(model, validloader, loss_fn)
        print(f'\tTrain Loss: {train_loss:.2f} - Valid Loss: {val_loss:.2f} - Valid Accuracy: {val_accuracy*100:.2f}%')
        if save_results:
            with open(res_file, 'a') as f:
                f.write(f'{epoch_idx},{train_loss:.2f},{val_loss:.2f},{val_accuracy*100:.2f}%\n')

    # save model
    torch.save(model.state_dict(), f'model-{curr_time}.pt')
    
    # get test accuracy
    model.eval()
    test_accuracy, test_loss = evaluate_model(model, testloader, loss_fn)
    print(f'Test accuracy: {test_accuracy} - Test loss: {test_loss}')


def train_one_epoch(
    trainloader: DataLoader, 
    model: nn.Module, 
    optimizer: torch.optim, 
    loss_fn: nn.modules.loss, 
    epoch_idx: int,
) -> float:
        '''
        Train a single epoch.
        '''
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

def evaluate_model(model: nn.Module, validloader: DataLoader, loss_fn: nn.modules.loss) -> float:
    '''
    Compute the accuracy and loss of the model on the validation set.
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

            loss = loss_fn(outputs, labels)
    return correct / total, loss