# License: BSD
# Author: Sasank Chilamkurthy (Lab Assistant)

from __future__ import print_function, division
from platform import architecture
from tkinter import Y
import torch
#from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
import Dataloader_NotJup
import wandb
import pandas as pd
from PIL import Image
from torch.utils.data import ConcatDataset 
import scipy.io
import timm

dataset_name = "food" #bird #dog, birddog, food, not working food and dog


# Training loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    example_ct = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0


            # Iterate over data.
            #for inputs, labels in dataloaders_one[phase]:
            for inputs, labels in dataloaders_one[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

               


            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                wandb.log({'epoch': epoch + 1, 'train_loss': epoch_loss, 'train_accuracy' : epoch_acc})

            if phase == 'val':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                wandb.log({'epoch': epoch + 1, 'val_loss': epoch_loss, 'val_accuracy' : epoch_acc})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


test = False 
epochs = 10
lr = 0.001

dict_data = {
"bird": 200,
"dog": 120,
"birddog": 320,
"food": 251
}

nb_classes = dict_data[dataset_name]

model_name = "mlpmixer"

wandb.login()

with wandb.init(project='convnext',
                config={ 
                    "learning_rate": lr,
                    "epochs": epochs,
                    "loss_function": "crossentropy",
                    "architecture": model_name,
                    "dataset": dataset_name,
                    
                }):

    config = wandb.config  
    # Data_loaders

    train_loader, test_loader, train_dataset, test_dataset = Dataloader_NotJup.load_dataset(dataset_name, test)
    dataloaders_one = {'train' :train_loader, 'val': test_loader} #for CUB called test

    image_datasets = {'train': train_dataset, 'val': test_dataset}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    
    len_dataset = dict_data[dataset_name]


    #Feature extrator
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    feature_extract = False

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        set_parameter_requires_grad(model_ft,feature_extract)
        model_ft.fc = nn.Linear(num_ftrs, len_dataset)
        input_size = 224
    
    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        set_parameter_requires_grad(model_ft,feature_extract)
        model_ft.classifier = nn.Linear(num_ftrs, len_dataset) 
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft,feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, len_dataset, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = len_dataset
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=True)
        num_ftrs = model_ft.classifier[6].in_features
        set_parameter_requires_grad(model_ft,feature_extract)
        model_ft.classifier[6] = nn.Linear(num_ftrs,len_dataset)
        input_size = 224

    elif model_name =='mlpmixer':
        #model_ft = timm.create_model('mixer_b16_224',pretrained=True, num_classes=len_dataset,drop_rate=0) no ft
        model_ft = timm.create_model('mixer_b16_224',pretrained=True,drop_rate=0) # si ft
        set_parameter_requires_grad(model_ft,feature_extract)
        model_ft.reset_classifier(len_dataset,'max')
        input_size = 224
        


    #number of layers
    #children = 0
    #for child in model_ft.children():
        #children += 1
    #print('Number of layers: '+str(children))

 

    
    model_ft = model_ft.to(device)

    #summary(model_ft, (3, 224, 224))
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Training
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)