import torch
import torchvision.transforms.functional as TF 
from torchvision import transforms, models
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from data_util import *
from util import device
from torchview import draw_graph
import time, os, fnmatch, shutil

class DeepNN(nn.Module):
    def __init__(self, dim_out=4, downsample=4):
        super(DeepNN, self).__init__()
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

class NueralNet(nn.Module):
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
 
def validate(model, val_loader):
    correct = 0
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch)        
        outputs = torch.argmax(y_pred, dim=1)
        correct += int(torch.sum(outputs==y_batch))
    return correct

def train_evaluate(task="pose", dim_out=4, useDeepNN=False, downsample=4, batch_size=8, lr=0.08, n_epochs = 50):
        if task == "pose":    
            task_reader=dataset_pose_task_loader
        elif task == "expression":
            task_reader=dataset_expression_task_loader
        elif task == "eyes":
            task_reader=dataset_eyes_task_loader

        if not useDeepNN:
            transform = transforms.Compose(
                    #[transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    [transforms.ToTensor()])
        else:
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])        
            
        # set up DataLoader for training set
        train_dataset = ImageTargetDataset("./data/", "trainset/all_train.list_"+str(downsample), task_reader=task_reader, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = ImageTargetDataset("./data/", "trainset/all_test1.list_"+str(downsample), task_reader=task_reader, transform=transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = ImageTargetDataset("./data/", "trainset/all_test2.list_"+str(downsample), task_reader=task_reader, transform=transform)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        FER_train_dataset, FER_test_dataset = getFERDataset()
        #train_val_dataset = ConcatDataset([train_dataset, val_dataset])
        train_val_dataset = ConcatDataset([train_dataset, FER_train_dataset, val_dataset])
        train_dataloader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
        #test_dataloader = torch.utils.data.DataLoader(FER_test_dataset, batch_size=batch_size, shuffle=True)

        # Train the model
        loss_fn = nn.CrossEntropyLoss()
        if not useDeepNN:
            model = NueralNet(dim_out=dim_out, downsample=downsample)
        else:
            model = DeepNN(dim_out=dim_out, downsample=downsample)

        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        model.train()
        train_loss = []
        for epoch in range(n_epochs):
            for X_batch, y_batch in train_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch %2 ==0:
                 print("epoch: ", epoch, "\t loss: ", loss.item())
                 train_loss.append(loss.item())
        plt.plot(train_loss)
        plt.savefig("train_loss.png")

        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        model_file_name = ("./models/model_" + timestamp)
        torch.save(model.state_dict(), model_file_name)
        torch.save(model, model_file_name + '.pt')

        correctness = validate(model, test_dataloader)
        test_accuracy = correctness/len(test_dataset)
        print("test accuracy: ", test_accuracy)

def draw_model():
    model_graph = draw_graph(NueralNet(dim_out=4), input_size=(8, 3, 30,32), expand_nested=True)
    model_graph.visual_graph

def show_image_grids():
    task_reader=dataset_pose_task_loader
    train_dataset = ImageTargetDataset("./data/", "trainset/all_train.list", task_reader=task_reader)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    imshow(torchvision.utils.make_grid(train_features.permute(0, 3, 1, 2), nrow=2))

def run_all_single_tasks(tasks):
    downsample = 4
    if 0 in tasks:
        print("------------------------------------------------")
        print("-----------   pose task    -----------------------")
        train_evaluate(task="pose", dim_out=4, downsample=downsample)

    if 1 in tasks:
        print("------------------------------------------------")
        print("-----------   expression task    -----------------------")
        train_evaluate(task="expression", dim_out=4, useDeepNN=True, downsample=downsample, batch_size=16, lr=0.5, n_epochs=200)

    if 2 in tasks:
        print("------------------------------------------------")
        print("-----------   eyes task    -----------------------")
        train_evaluate(task="eyes", dim_out=2, downsample=downsample)

if __name__ == '__main__':
    run_all_single_tasks(tasks=[1])        
    #run_all_single_tasks(tasks=[1])        
    #show_image_grids()
