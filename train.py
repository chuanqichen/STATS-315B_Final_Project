import torch
import torchvision.transforms.functional as TF 
from torchvision import transforms, models
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import numpy as np
from data_util import *
from model import *
from util import device
from torchview import draw_graph
import time, os, fnmatch, shutil, copy
import warnings
warnings.filterwarnings("ignore")

def validate(model, val_loader, useNueralNetsEnsemble=False):
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)        
            if not useNueralNetsEnsemble:               
                    outputs = torch.argmax(y_pred, dim=1)
            else:
                    votes = y_pred.argmax(2)
                    #outputs = torch.mode(votes, 1)
                    outputs = torch.Tensor([max(vote, key=vote.tolist().count).item() for vote in votes]).to(device)
            
            correct += torch.mean((outputs==y_batch).float())           
    return correct.item()/len(val_loader)

def train_evaluate(task="pose", dim_out=4, useDeepNN=False, useNueralNetsEnsemble=False, useFER=False, downsample=4, batch_size=8, lr=0.08, n_epochs = 50):
        if task == "pose":    
            task_reader=dataset_pose_task_loader
        elif task == "expression":
            task_reader=dataset_expression_task_loader
        elif task == "eyes":
            task_reader=dataset_eyes_task_loader

        if not useDeepNN:
            transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    #[transforms.ToTensor(), transforms.RandomHorizontalFlip()])
        else:
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])        
            
        # set up DataLoader for training set
        train_dataset = ImageTargetDataset("./data/", "trainset/all_train.list_"+str(downsample), task_reader=task_reader, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_dataset = ImageTargetDataset("./data/", "trainset/all_test1.list_"+str(downsample), task_reader=task_reader, transform=transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_dataset = ImageTargetDataset("./data/", "trainset/all_test2.list_"+str(downsample), task_reader=task_reader, transform=transform)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        
        if useFER:
            FER_train_dataset, FER_val_dataset, FER_test_dataset = getFERDataset()
            #train_combined_dataset = ConcatDataset([train_dataset, FER_train_dataset])
            #train_dataloader = torch.utils.data.DataLoader(train_combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            train_dataloader = torch.utils.data.DataLoader(FER_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_dataloader = torch.utils.data.DataLoader(FER_val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            test_dataloader = torch.utils.data.DataLoader(FER_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Train the model
        loss_fn = nn.CrossEntropyLoss()
        if not useDeepNN and not useNueralNetsEnsemble:
            model = NueralNet(dim_out=dim_out, downsample=downsample)
        elif  not useDeepNN  and useNueralNetsEnsemble:
            model = NueralNetsEnsemble(dim_out=dim_out, downsample=downsample)
        else:
            model = DeepNN(dim_out=dim_out, downsample=downsample)

        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.15)
        model.train()
        train_loss = []
        train_acc = []
        val_acc = []
        best_acc = 0
        for epoch in range(n_epochs):
            acc = 0
            for X_batch, y_batch in train_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                if not useNueralNetsEnsemble:               
                    loss = loss_fn(y_pred, y_batch)
                    outputs = torch.argmax(y_pred, dim=1)
                else:
                    loss = sum([loss_fn(y_pred[:, i,:], y_batch) for i in range(y_pred.shape[1])])/y_pred.shape[1]
                    votes = y_pred.argmax(2)
                    #outputs = torch.mode(votes, 1)
                    outputs = torch.Tensor([max(vote, key=vote.tolist().count).item() for vote in votes]).to(device)
                acc += torch.mean((outputs==y_batch).float()).item()
                loss.backward()
                optimizer.step()

            if epoch %10 ==0:
                 train_loss.append((epoch, loss.item()))
                 train_acc.append((epoch, acc/len(train_dataloader)))
                 val_accuracy = validate(model, val_dataloader,useNueralNetsEnsemble)
                 val_acc.append((epoch, val_accuracy))
                 print("epoch: ", epoch, "\t train loss: ", loss.item(), "\t acc: ", train_acc[-1], "\t val acc: ", val_accuracy)
                 if val_accuracy > best_acc:
                     best_acc = val_accuracy
                     best_model = copy.deepcopy(model)
            exp_lr_scheduler.step()

        x, y = map(list, zip(*train_loss))
        plt.plot(x, y)
        plt.title("Training loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig("train_loss.png")
        plt.figure() 
        x, y = map(list, zip(*train_acc))
        plt.plot(x, y,  label="train acc")
        x, y = map(list, zip(*val_acc))
        plt.plot(x, y, label='val acc"')
        plt.legend(loc="upper right")
        plt.title("Traing and evaluation Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.savefig("train_acc.png")

        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        model_file_name = ("./models/best_model_" + timestamp)
        torch.save(best_model.state_dict(), model_file_name)
        torch.save(best_model, model_file_name + '.pt')

        test_accuracy = validate(best_model, test_dataloader,useNueralNetsEnsemble)
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
    downsample = 1
    if 0 in tasks:
        print("------------------------------------------------")
        print("-----------   pose task    -----------------------")
        train_evaluate(task="pose", dim_out=4, downsample=downsample)

    if 1 in tasks:
        print("------------------------------------------------")
        print("-----------   expression task    -----------------------")
        #train_evaluate(task="expression", dim_out=4, useDeepNN=False, downsample=downsample, batch_size=16, lr=0.5, n_epochs=200)
        #train_evaluate(task="expression", dim_out=4, useDeepNN=False, useFER=False,
        #               downsample=downsample, batch_size=16, lr=0.01, n_epochs=10)
        #train_evaluate(task="expression", dim_out=4, useDeepNN=True, useFER=True,
        #               downsample=downsample, batch_size=16, lr=0.001, n_epochs=100)
        train_evaluate(task="expression", dim_out=4, useDeepNN=False, useNueralNetsEnsemble=True, useFER=False,
                       downsample=downsample, batch_size=16, lr=0.01, n_epochs=200)
 
    if 2 in tasks:
        print("------------------------------------------------")
        print("-----------   eyes task    -----------------------")
        train_evaluate(task="eyes", dim_out=2, downsample=downsample)

if __name__ == '__main__':
    run_all_single_tasks(tasks=[1])        
    #run_all_single_tasks(tasks=[1])        
    #show_image_grids()
