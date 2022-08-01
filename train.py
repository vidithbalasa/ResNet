import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from params import Params
from typing import Optional
from tqdm import tqdm
from datetime import datetime
from data_loader import get_CIFAR_data

def train(model: nn.Module, epochs: int, save_name: str) -> None:
    '''
    Train the model.
        @param model: the model to train
        @param epochs: the number of epochs to train for
        @param save_name: where to save the model - full path and filename without extension (e.g. '.csv')
    '''
    trainloader, testloader, classes = get_CIFAR_data()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Params.LEARNING_RATE)
    if save_name:
        with open(f'{save_name}.csv', 'w') as f:
            f.write(f'epoch,train_loss,train_accuracy,test_loss,test_accuracy,test_top_5_accuracy\n')
        with open(f'{save_name}_params.csv', 'w') as f:
            params_to_save = {
                'learning_rate': Params.LEARNING_RATE,
                'epochs': epochs,
                'batch_size': Params.BATCH_SIZE,
                'device': Params.DEVICE,
                'num_classes': Params.NUM_CLASSES
            }
            # save the dict as a csv
            ordered_keys = [str(x) for x in params_to_save.keys()]
            ordered_vals = [str(params_to_save[x]) for x in ordered_keys]
            f.write(','.join(ordered_keys) + '\n')
            f.write(','.join(ordered_vals))
    
    last_loss = float('inf')
    for epoch_idx in range(epochs):
        model.train(True)
        train_loss, train_accuracy = train_one_epoch(model, trainloader, optimizer, loss_fn, epoch_idx)
        model.eval()
        test_loss, test_accuracy, test_top_5_acc = evaluate_model(model, testloader, loss_fn, include_top_5=True)
        print(f'\tTrain Loss: {train_loss:.2f} - Train Accuracy: {train_accuracy:.2f}%, \
            Test Loss: {test_loss:.2f} - Test Accuracy: {test_accuracy*100:.2f}% - Test Top 5 Accuracy: {test_top_5_acc*100:.2f}%')
        if save_name:
            with open(f'{save_name}.csv', 'a') as f:
                f.write(f'{epoch_idx},{train_loss:.2f},{train_accuracy:.2f}%,{test_loss:.2f},{test_accuracy*100:.2f}%,{test_top_5_acc*100:.2f}%\n')
        # save 
        if train_loss < last_loss:
            torch.save(model.state_dict(), f'{save_name}.pt')
            last_loss = train_loss
            print(f'Saved model from epoch {epoch_idx}')

    # save model
    torch.save(model.state_dict(), f'{save_name}_final.pt')
    print(f'Model saved to {save_name}_final.pt')


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
        total = 0

        # tqdm loader with title 'epoch {epoch_idx}'
        for batch_idx, (images, labels) in tqdm(enumerate(trainloader), desc=f'Epoch {epoch_idx}'):
            images, labels = images.to(Params.DEVICE), labels.to(Params.DEVICE)
            # zero gradients for each batch
            optimizer.zero_grad()
            # get model predictions
            outputs = model(images)
            # check if the model is correct
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # compute loss & gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
            # backpropagate
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(trainloader), correct / total

def evaluate_model(model: nn.Module, validloader: DataLoader, loss_fn: nn.modules.loss, include_top_5: bool=False) -> float:
    '''
    Compute the accuracy and loss of the model on the validation set.
    '''
    correct = 0
    top_5_correct = 0
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

            if include_top_5:
                # compute top 5 accuracy
                _, top_5_pred = torch.topk(outputs.data, 5)
                top_5_correct += np.array([1 for i in range(top_5_pred.size(0)) if labels[i] in top_5_pred[i]]).sum()

            # compute loss
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
    valid_loss = running_loss / len(validloader)
    valid_accuracy = correct / total
    valid_top_5_accuracy = top_5_correct / total
    if include_top_5:
        return valid_loss, valid_accuracy, valid_top_5_accuracy
    return valid_loss, valid_accuracy