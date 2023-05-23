import torch
import torchvision.transforms.functional as TF 
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from data_util import *
from util import device
 
class NueralNet(nn.Module):
    def __init__(self, dim_out=4):
        super(NueralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=25, kernel_size=3, padding=1),
            nn.Dropout(),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(25*15*16, dim_out)
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

def train_evaluate(task="pose", dim_out=4):
        if task == "pose":    
            task_reader=dataset_pose_task_loader
        elif task == "expression":
            task_reader=dataset_expression_task_loader
        elif task == "eyes":
            task_reader=dataset_eyes_task_loader

        # set up DataLoader for training set
        train_dataset = ImageTargetDataset("./data/", "trainset/all_train.list", task_reader=task_reader, transform=ToTensor())
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataset = ImageTargetDataset("./data/", "trainset/all_test1.list", task_reader=task_reader, transform=ToTensor())
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)
        test_dataset = ImageTargetDataset("./data/", "trainset/all_test2.list", task_reader=task_reader, transform=ToTensor())
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
        
        train_val_dataset = ConcatDataset([train_dataset, val_dataset])
        train_dataloader = torch.utils.data.DataLoader(train_val_dataset, batch_size=8, shuffle=True)

        # Train the model
        n_epochs = 50
        loss_fn = nn.CrossEntropyLoss()
        model = NueralNet(dim_out=dim_out)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.08)
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
            if epoch %10 ==0:
                 train_loss.append(loss.item())
        plt.plot(train_loss)
        plt.savefig("train_loss.png")

        correctness = validate(model, test_dataloader)
        test_accuracy = correctness/len(test_dataset)
        print("test accuracy: ", test_accuracy)

if __name__ == '__main__':
	print("------------------------------------------------")
	print("-----------   pose task    -----------------------")
	train_evaluate(task="pose", dim_out=4)

	print("------------------------------------------------")
	print("-----------   expression task    -----------------------")
	train_evaluate(task="expression", dim_out=4)

	print("------------------------------------------------")
	print("-----------   eyes task    -----------------------")
	train_evaluate(task="eyes", dim_out=2)
