import numpy as np
import torch.utils.data as data
import torch
import torchvision
from torchvision import transforms, models, datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import os
import cv2
import matplotlib.pyplot as plt
from util import device
import warnings
warnings.filterwarnings("ignore")

# pose: straight,left, right, up
def dataset_pose_task_loader(folder, fileslist):
    full_path_file_list = os.path.join(folder, fileslist)
    file_targets = []
    with open(full_path_file_list, 'r') as file:
        for line in file.readlines():
            line = line.strip('\n')[1:]
            if "left" in line:
                target = 0 #np.array([1, 0, 0, 0])
            elif "right" in line:
                target = 1 #np.array([0, 1, 0, 0])
            elif "straight" in line:
                target = 2 #np.array([0, 0, 1, 0] )
            else:  #"up"
                target = 3 #np.array([0, 0, 0, 1])

            face_image_file = os.path.join(folder, line)
            file_targets.append( (face_image_file, target) )				
    return file_targets

# neutral, happy, sad, angry
#0=Angry, 3=Happy, 4=Sad, 6=Neutral in FER dataset 
# class_to_idx: {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}
def dataset_expression_task_loader(folder, fileslist):
    full_path_file_list = os.path.join(folder, fileslist)
    file_targets = []
    with open(full_path_file_list, 'r') as file:
        for line in file.readlines():
            line = line.strip('\n')[1:]
            if "sad" in line:
                target = 3 
            elif "neutral" in line:
                target = 2 
            elif "happy" in line:
                target = 1 
            else:  #"angry"
                target = 0 

            face_image_file = os.path.join(folder, line)
            file_targets.append( (face_image_file, target) )				
    return file_targets

#eyes: open, sunglasses
def dataset_eyes_task_loader(folder, fileslist):
    full_path_file_list = os.path.join(folder, fileslist)
    file_targets = []
    with open(full_path_file_list, 'r') as file:
        for line in file.readlines():
            line = line.strip('\n')[1:]
            if "open" in line:
                target = 0
            elif "sunglasses" in line:
                target = 1 

            face_image_file = os.path.join(folder, line)
            file_targets.append( (face_image_file, target) )				
    return file_targets

class ImageTargetDataset(data.Dataset):
	def __init__(self, root, dataset_list, transform=None, target_transform=None,
			task_reader=dataset_pose_task_loader):
		self.root   = root
		self.file_targets = task_reader(root, dataset_list)		
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		image_file, target = self.file_targets[index]
		image = cv2.imread(image_file)
		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
        
		return image, target

	def __len__(self):
		return len(self.file_targets)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# FER-2013 The Facial Expression Recognition 2013 (FER-2013) Dataset
# https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge
def getFERDataset(image_size=224):
     # Creating the train/test dataloaders from images
    FER_root_dir = './data/FER-2013/images'
    transform = transforms.Compose([transforms.RandomResizedCrop(image_size),transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    total_dataset = datasets.ImageFolder(FER_root_dir, transform)

    train_size = int(0.8 * len(total_dataset))
    test_size = len(total_dataset) - train_size
    FER_train_dataset, FER_test_dataset = torch.utils.data.random_split(total_dataset, [train_size, test_size])
    FER_val_dataset, FER_test_dataset = torch.utils.data.random_split(FER_test_dataset, [test_size//2, test_size-test_size//2])

    FER_train_dataloader = torch.utils.data.DataLoader(FER_train_dataset, batch_size=10, shuffle=True, num_workers=4)
    FER_test_dataloader = torch.utils.data.DataLoader(FER_test_dataset, batch_size=10, shuffle=True, num_workers=4)

    class_names = total_dataset.classes
    print(class_names)
    print(total_dataset.class_to_idx)
    num_classes = len(class_names)
    return FER_train_dataset, FER_val_dataset, FER_test_dataset

if __name__ == '__main__':      
      train_dataset = ImageTargetDataset("./data/", "trainset/straighteven_train.list")
      train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
      train_features, train_labels = next(iter(train_dataloader))
      print(f"Feature batch shape: {train_features.size()}")
      print(f"Labels batch shape: {train_labels.size()}")
      img = train_features[0].squeeze()
      label = train_labels[0]
      #plt.imshow(img)
      plt.title = label 
      plt.show()
      print(f"Label: {label}")
      #imshow(torchvision.utils.make_grid(train_features.permute(0, 3, 1, 2), nrow=2))
      print(f"Label: {label}")

      FER_train_dataset, FER_test_dataset = getFERDataset()
      train_val_dataset = ConcatDataset([train_dataset, FER_train_dataset])
      train_dataloader = torch.utils.data.DataLoader(train_val_dataset, batch_size=8, shuffle=True)
      train_features, train_labels = next(iter(train_dataloader))
      print(f"Feature batch shape: {train_features.size()}")
      print(f"Labels batch shape: {train_labels.size()}")
      img = train_features[0].squeeze()
      label = train_labels[0]
      plt.imshow(img.moveaxis(0, -1))
      plt.show()
      plt.title = label 
