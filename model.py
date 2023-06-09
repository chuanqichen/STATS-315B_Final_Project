from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import numpy as np
from data_util import *
from util import device
from torchview import draw_graph
import time, os, fnmatch, shutil, copy
import warnings
warnings.filterwarnings("ignore")

class DeepNN0(nn.Module):
    def __init__(self, dim_out=4, downsample=4):
        super(DeepNN0, self).__init__()
        pretrain_model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        num_features = pretrain_model.fc.in_features
        pretrain_model.fc = nn.Linear(num_features, 20)

        size_scale = int(4/downsample)**2

        self.model = nn.Sequential(
            pretrain_model,
            nn.ReLU(), 
             nn.Linear(20, dim_out)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class DeepNN1(nn.Module):
    def __init__(self, dim_out=4, downsample=4):
        super(DeepNN1, self).__init__()
        pretrain_model = models.resnet50(weights=models.ResNet50_Weights)
        num_features = pretrain_model.fc.in_features
        pretrain_model.fc = nn.Linear(num_features, dim_out)

        size_scale = int(4/downsample)**2

        self.model = nn.Sequential(
            pretrain_model,
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DeepNN2(nn.Module):
    def __init__(self, dim_out=4, downsample=4):
        super(DeepNN2, self).__init__()
        pretrain_model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        num_features = pretrain_model.fc.in_features
        pretrain_model.fc = nn.Linear(num_features, 512)

        size_scale = int(4/downsample)**2

        self.model = nn.Sequential(
            pretrain_model,
            nn.ReLU(), 
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.25),
            nn.Linear(128, dim_out)
        )

    def forward(self, x):
        x = self.model(x)
        return x

DeepNN = DeepNN1

class NueralNetsEnsemble(nn.Module):
    def __init__(self, dim_out=4, downsample=4):
        super(NueralNetsEnsemble, self).__init__()
        size_scale = int(4/downsample)**2
        self.model =  nn.ModuleList()
        for _ in range(4):
            self.model.append( nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
                nn.Dropout(0.50),
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(128*15*16*size_scale, 256),
                nn.ReLU(), 
                nn.Dropout(0.25),
                nn.Linear(256, dim_out)
            ))
        for _ in range(3):
            self.model.append( nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
                nn.Dropout(0.25),
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64*15*16*size_scale, 128),
                nn.ReLU(), 
                nn.Dropout(0.25),
                nn.Linear(128, dim_out)
            ))            

    def forward(self, x):
        outputs = []
        for model in self.model:
            outputs.append(model(x))
        outputs = torch.stack(outputs)
        return outputs.swapaxes(0, 1)


class NueralNet(nn.Module):
    def __init__(self, dim_out=4, downsample=4):
        super(NueralNet, self).__init__()
        size_scale = int(4/downsample)**2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128*15*16*size_scale, 256),
            nn.ReLU(), 
            nn.Dropout(0.25),
            nn.Linear(256, dim_out)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class NueralNet0(nn.Module):
    def __init__(self, dim_out=4, downsample=4):
        super(NueralNet, self).__init__()
        size_scale = int(4/downsample)**2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=25, kernel_size=3, padding=1),
            nn.Dropout(),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(25*15*16*size_scale, dim_out)
        )

    def forward(self, x):
        x = self.model(x)
        return x
