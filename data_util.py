import numpy as np
import torch.utils.data as data
import torch
import torchvision
import os
import cv2
import matplotlib.pyplot as plt
from util import device

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
def dataset_expression_task_loader(folder, fileslist):
    full_path_file_list = os.path.join(folder, fileslist)
    file_targets = []
    with open(full_path_file_list, 'r') as file:
        for line in file.readlines():
            line = line.strip('\n')[1:]
            if "neutral" in line:
                target = 0 
            elif "happy" in line:
                target = 1 
            elif "sad" in line:
                target = 2 
            else:  #"angry"
                target = 3 

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
		return np.array(image), target

	def __len__(self):
		return len(self.file_targets)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == '__main__':
      dataset = ImageTargetDataset("./data/", "trainset/straighteven_train.list")
      train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
      train_features, train_labels = next(iter(train_dataloader))
      print(f"Feature batch shape: {train_features.size()}")
      print(f"Labels batch shape: {train_labels.size()}")
      img = train_features[0].squeeze()
      label = train_labels[0]
      plt.imshow(img)
      plt.title = label 
      plt.show()
      print(f"Label: {label}")
      imshow(torchvision.utils.make_grid(train_features.permute(0, 3, 1, 2), nrow=2))
      print(f"Label: {label}")
