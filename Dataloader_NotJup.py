from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import scipy.io
from PIL import Image
from torch.utils.data import ConcatDataset 
import pandas as pd

class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1].split('/')[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
#         import pdb
#         pdb.set_trace()
        return self.split_info[x] == self.split

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content

class DOGDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for DOG Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}splits/file_list.mat")
        image_files = [o[0][0].split('/')[1] for o in image_info]
        
        split_info = self.get_file_content(f"{image_root_path}/splits/{split}_list.mat")
        split_files = [o[0][0].split('/')[1] for o in split_info]
        self.split_info = {}
        if split == 'train' :
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "1"
                else:
                    self.split_info[image] = "0"
        elif split== 'test' :
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "0"
                else:
                    self.split_info[image] = "1"
                    
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(DOGDataset, self).__init__(root=f"{image_root_path}Images", is_valid_file = self.is_valid_file,
                                         *args, **kwargs)
        
        ## modify class index as we are going to concat to first dataset
        self.class_to_idx = {class_: idx+200 for idx, class_ in enumerate(self.class_to_idx)}
        
    def is_valid_file(self, x):
        return self.split_info[x] == self.split
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(os.path.join(path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        ## modify target class index as we are going to concat to first dataset
        return img, target + 200

    @staticmethod
    def get_file_content(file_path):
        content =  scipy.io.loadmat(file_path)
        return content['file_list']

class FOODDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))        
        ])
        return (
            data_transform(Image.open(row["path"])), row['label']
        )

def print_stat(train_dataset, test_dataset, train_loader, test_loader, dataset_name, test):
    #Dataset Type
    print("Dataset loaded type:", dataset_name)

    if test: 
        #Dataset Statistics
        print('Number of train samples:', len(train_dataset))
        print('Number of test samples:', len(test_dataset))

        #Loader Statistics
        print("Len train loader:", len(train_loader))
        print("Len test loader:", len(test_loader))

        for i, (inputs, labels) in enumerate(train_loader):
            print("Inputs in train loader:", inputs.shape)
            print("Labels in train loader:",labels)
            print('='*50)
            break

def load_dataset(dataset_name, test = True): #bird #dog, birddog, food,

    # Set train and test set
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Write data transform here as per the requirement
    data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if dataset_name == "bird":
        data_root = "/apps/local/shared/CV703/datasets/CUB/CUB_200_2011/"
        train_dataset_cub= CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
        train_loader_cub = torch.utils.data.DataLoader(train_dataset_cub, batch_size=32, drop_last=True, shuffle=True)
        test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=32, drop_last=False, shuffle=False)
        print_stat(train_dataset_cub, test_dataset_cub, train_loader_cub, test_loader_cub, dataset_name, test)
        return train_loader_cub, test_loader_cub, train_dataset_cub, test_dataset_cub

    elif dataset_name == "dog":
        data_root = "/apps/local/shared/CV703/datasets/dog/"
        train_dataset_dog = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset_dog = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
        train_loader_dog = torch.utils.data.DataLoader(train_dataset_dog, batch_size=32, drop_last=True, shuffle=True)
        test_loader_dog = torch.utils.data.DataLoader(test_dataset_dog, batch_size=32, drop_last=False, shuffle=False)
        print_stat(train_dataset_dog, test_dataset_dog, train_loader_dog, test_loader_dog, dataset_name, test)
        return train_loader_dog, test_loader_dog, train_dataset_dog, test_dataset_dog

    elif dataset_name == "birddog":
        #CUB
        data_root = "/apps/local/shared/CV703/datasets/CUB/CUB_200_2011/"
        train_dataset_cub= CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
        train_loader_cub = torch.utils.data.DataLoader(train_dataset_cub, batch_size=32, drop_last=True, shuffle=True)
        test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=32, drop_last=False, shuffle=False)
        print_stat(train_dataset_cub, test_dataset_cub, train_loader_cub, test_loader_cub, dataset_name, test)
        #DOG
        data_root = "/apps/local/shared/CV703/datasets/dog/"
        train_dataset_dog = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset_dog = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
        train_loader_dog = torch.utils.data.DataLoader(train_dataset_dog, batch_size=32, drop_last=True, shuffle=True)
        test_loader_dog = torch.utils.data.DataLoader(test_dataset_dog, batch_size=32, drop_last=False, shuffle=False)
        print_stat(train_dataset_dog, test_dataset_dog, train_loader_dog, test_loader_dog, dataset_name, test)

        concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_dog])

        concat_loader_train = torch.utils.data.DataLoader(
                        concat_dataset_train,
                        batch_size=128, shuffle=True,
                        num_workers=1, pin_memory=True
                        )

        concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_dog])

        concat_loader_test = torch.utils.data.DataLoader(
                        concat_dataset_test,
                        batch_size=128, shuffle=False,
                        num_workers=1, pin_memory=True
                        )

        if test: 
            print(concat_dataset_train.datasets)
            print(concat_dataset_train.datasets[0].class_to_idx)
            print(concat_dataset_train.datasets[1].class_to_idx)

            for i, (inputs, labels) in enumerate(concat_loader_train):
                print(inputs.shape)
                print(labels)
                print('='*50)
                break

        return concat_loader_train, concat_loader_test, concat_dataset_train, concat_dataset_test

    elif dataset_name == "food":
        data_dir = "/apps/local/shared/CV703/datasets/FoodX/food_dataset"

        split = 'train'
        train_df = pd.read_csv(f'{data_dir}/annot/{split}_info.csv', names= ['image_name','label'])
        train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))

        split = 'val'
        val_df = pd.read_csv(f'{data_dir}/annot/{split}_info.csv', names= ['image_name','label'])
        val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))
            
        train_dataset_food = FOODDataset(train_df)
        test_dataset_food = FOODDataset(val_df)

        train_loader_food = torch.utils.data.DataLoader(train_dataset_food, batch_size=32, drop_last=True, shuffle=True)
        test_loader_food = torch.utils.data.DataLoader(test_dataset_food, batch_size=32, drop_last=False, shuffle=True)
        print_stat(train_dataset_food, test_dataset_food, train_loader_food, test_loader_food, dataset_name, test)

        #print("LEN:", train_dataset_food.dataframe.count(axis=0, level=None, numeric_only=False))
        return train_loader_food, test_loader_food, train_dataset_food, test_dataset_food


    else: 
        print("No dataset found !!!!")