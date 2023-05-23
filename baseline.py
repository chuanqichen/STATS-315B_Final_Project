import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from data_util import *

def train_evaluate(task="pose"):
	if task == "pose":    
		task_reader=dataset_pose_task_loader
	elif task == "expression":
		task_reader=dataset_expression_task_loader
	elif task == "eyes":
		task_reader=dataset_eyes_task_loader

	train_dataset = ImageTargetDataset("./data/", "trainset/all_train.list", task_reader=task_reader)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
	test1_dataset = ImageTargetDataset("./data/", "trainset/all_test1.list", task_reader=task_reader)
	test1_dataloader = torch.utils.data.DataLoader(test1_dataset, batch_size=len(test1_dataset), shuffle=True)
	test2_dataset = ImageTargetDataset("./data/", "trainset/all_test2.list", task_reader=task_reader)
	test2_dataloader = torch.utils.data.DataLoader(test2_dataset, batch_size=len(test2_dataset), shuffle=True)

	X_train, y_train = next(iter(train_dataloader))
	X_test1, y_test1 = next(iter(test1_dataloader))
	X_train = torch.vstack((X_train, X_test1))
	y_train = torch.cat((y_train, y_test1))

	X_test, y_test = next(iter(test2_dataloader))
	X_train = X_train.reshape(X_train.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)

	# GAUSSIAN NAIVE BAYES
	gnb = GaussianNB()
	# train the model
	gnb.fit(X_train, y_train)
	# make predictions
	gnb_pred = gnb.predict(X_test)
	# print the accuracy
	print("Accuracy of Gaussian Naive Bayes: ",
		accuracy_score(y_test, gnb_pred))

	# DECISION TREE CLASSIFIER
	dt = DecisionTreeClassifier(random_state=0)
	# train the model
	dt.fit(X_train, y_train)
	# make predictions
	dt_pred = dt.predict(X_test)
	# print the accuracy
	print("Accuracy of Decision Tree Classifier: ",
		accuracy_score(y_test, dt_pred))

	# SUPPORT VECTOR MACHINE
	svm_clf = svm.SVC(kernel='linear') # Linear Kernel
	# train the model
	svm_clf.fit(X_train, y_train)
	# make predictions
	svm_clf_pred = svm_clf.predict(X_test)
	# print the accuracy
	print("Accuracy of Support Vector Machine: ",
		accuracy_score(y_test, svm_clf_pred))
	

if __name__ == '__main__':
	print("------------------------------------------------")
	print("-----------   pose task    -----------------------")
	train_evaluate(task="pose")

	print("------------------------------------------------")
	print("-----------   expression task    -----------------------")
	train_evaluate(task="expression")

	print("------------------------------------------------")
	print("-----------   eyes task    -----------------------")
	train_evaluate(task="eyes")
