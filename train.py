import torch
import torch.nn as nn
from torchvision import transforms
from main import create_resnet
from params import Params
import fastai.vision.all as fastai
import os


def train(epochs: int):
    '''
    Train the model.
    '''
    model = create_resnet('resnet50', num_classes=10)
    device = torch.device('mps')
    model.to(device)

    # Load the data
    data_path = fastai.untar_data(fastai.URLs.IMAGENETTE_160)

    data_block = fastai.DataBlock(
        blocks=(fastai.ImageBlock, fastai.CategoryBlock),
        get_items=fastai.get_image_files,
        splitter=fastai.RandomSplitter(),
        get_y=fastai.parent_label,
        item_tfms=fastai.Resize(460),
        batch_tfms=fastai.aug_transforms(size=224, min_scale=0.75),
    )
    train_path = os.path.join(data_path, 'train')
    data_loader = data_block.dataloaders(train_path, bs=Params.BATCH_SIZE)

    learner = fastai.Learner(data_loader, model, loss_func=fastai.CrossEntropyLossFlat(), metrics=fastai.accuracy)
    learner.fit(epochs=epochs)
