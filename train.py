import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from params import Params
from typing import Optional
from tqdm import tqdm
from datetime import datetime
from data_loader import get_CIFAR_data

def train(model: nn.Module, save_name: str, top_k: int=0) -> None:
    '''
    Train the model.
        @param model: the model to train
        @param epochs: the number of epochs to train for
        @param save_name: where to save the model - full path and filename without extension (e.g. '.csv')
    '''
    trainloader, testloader, _ = get_CIFAR_data()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Params.LEARNING_RATE, momentum=Params.MOMENTUM, weight_decay=Params.WEIGHT_DECAY)
    if save_name:
        with open(f'{save_name}.csv', 'w') as f:
            cols = 'epoch,train_loss,train_accuracy,valid_loss,valid_accuracy'
            f.write(cols + ',valid_top_k_accuracy' if top_k > 0 else cols)
        Params.save_params_to_csv(f'{save_name}_params.csv')
    
    last_loss = float('inf')
    for epoch_idx in range(Params.EPOCHS):
        model.train(True)
        train_loss, train_accuracy = train_one_epoch(model, trainloader, optimizer, loss_fn, epoch_idx)
        model.eval()
        test_loss, test_accuracy, *test_top_k_acc = evaluate_model(model, testloader, loss_fn, top_k)
        print(f'\tTrain Loss: {train_loss:.2f} - Train Accuracy: {train_accuracy*100:.2f}%, \
            Test Loss: {test_loss:.2f} - Test Accuracy: {test_accuracy*100:.2f}%')
        if save_name:
            row = f'\n{epoch_idx},{train_loss:.2f},{train_accuracy*100:.2f}%,{test_loss:.2f},{test_accuracy*100:.2f}%'
            with open(f'{save_name}.csv', 'a') as f:
                f.write(row + f',{test_top_k_acc:.2f}%' if top_k > 0 else row)
        # save 
        if test_loss < last_loss:
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
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch_idx}'):
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

def evaluate_model(model: nn.Module, validloader: DataLoader, loss_fn: nn.modules.loss, top_k: int=0) -> float:
    '''
    Compute the accuracy and loss of the model on the validation set.
    '''
    correct = 0
    top_k_corrent = 0
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

            if top_k > 0:
                # compute top 5 accuracy
                _, top_k_pred = torch.topk(outputs.data, top_k)
                top_k_corrent += np.array([1 for i in range(top_k_pred.size(0)) if labels[i] in top_k_pred[i]]).sum()

            # compute loss
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
    valid_loss = running_loss / len(validloader)
    valid_accuracy = correct / total
    valid_top_k_acc = top_k_corrent / total
    if top_k:
        return valid_loss, valid_accuracy, valid_top_k_acc
    return valid_loss, valid_accuracy