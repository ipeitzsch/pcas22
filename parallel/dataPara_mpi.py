# Normal imports
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
import csv
import time

import medmnist
from medmnist import INFO, Evaluator

# MPI
from mpi4py import MPI

# Simple CNN model
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def sublistFromDataclass(d, s, end):
	lst = []
	for i in range(s, end):
		lst.append(d[i])

	return lst

if __name__ == "__main__":
	# Arg parser
	parser = argparse.ArgumentParser(description='Run neural network using data parallel on multiple GPUs')
	parser.add_argument('-f', dest='f', help='File containing description of neural network')
	parser.add_argument('--num', dest='n', type=int, help='Number of nodes to use', default=1)
	parser.add_argument('-i', dest='i', type=int, help='Number of test images', default=7180)
	parser.add_argument('-g', dest='g', action='store_true', help='Use GPUs')
	#parser.add_argument('-m', dest='m', type=int, help='Number of GPUs per node to use', default=1)
	args = parser.parse_args()

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	# Get data
	data_flag = 'pathmnist'
	
	info = INFO[data_flag]
	task = info['task']
	n_channels = info['n_channels']
	n_classes = len(info['label'])
	DataClass = getattr(medmnist, info['python_class'])
	
	# Preprocessing
	data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
	
	# Get data
	train = DataClass(split='train', transform=data_transform, download=True)
	test = DataClass(split='test', transform=data_transform, download=True)
	#train = data.DataLoader(dataset=train, batch_size=128, shuffle=True)
	#test = data.DataLoader(dataset=test, batch_size=256, shuffle=False)

	# Create larger dataset, if necessary
	while args.i > len(test):
		# Makes a list longer than needed, but doesn't matter
		test = test + test

	model = Net(in_channels=n_channels, num_classes=n_classes)

	# Train if no parameters provided
	if type(args.f) == type(None):
		if task == "multi-label, binary-class":
			criterion = nn.BCEWithLogitsLoss()
		else:
			criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
		for epoch in range(3):
			train_correct = 0
			train_total = 0
			test_correct = 0
			test_total = 0
			
			model.train()
			for inputs, targets in tqdm(train):
				#forward + backward + optimize
				optimizer.zero_grad()
				outputs = model(inputs)
				
				if task == 'multi-label, binary-class':
					targets = targets.to(torch.float32)
					loss = criterion(outputs, targets)
				else:
					targets = targets.squeeze().long()
					loss = criterion(outputs, targets)
				
				loss.backward()
				optimizer.step()
		torch.save(model, 'model.pt')
	else:
		model = torch.load(args.f)
		torch.save(model, 'model.pt')

	# Start timer
	if rank == 0:
		startTime = time.time()

	# Now, distribute images
	if rank == 0:
		inc = int(args.i/args.n)
		mod = args.i%args.n
		start = 0
		nxt = inc
		if mod > 0:
			nxt = nxt + 1
		myData = sublistFromDataclass(test, start, nxt)
		start = nxt
		for i in range(1,args.n):
			nxt = start + inc
			if mod > i:
				nxt += 1
			comm.send(sublistFromDataclass(test, start, nxt), dest=i, tag=0)
			start = nxt
	else:
		myData = comm.recv(source=0, tag=0)

	# Inference time
	res = []
	batchSize = 256
	testLoad = data.DataLoader(myData, batch_size=batchSize, shuffle=False)

	with torch.no_grad():
		if args.g:
			if torch.cuda.is_available():
				#if torch.cuda.device_count() < args.m:
				#	print("ERROR: not enough GPUs on this node")
				#	quit()
				
				model = model.cuda()
			else:
				print("ERROR: Could not use CUDA")
				quit()
		for inputs, targets in testLoad:
			if args.g:
				res.append(model(inputs.cuda()).softmax(dim=-1))
			else:
				res.append(model(inputs).softmax(dim=-1))

	# Recover all inferences at root
	count = 0
	if rank == 0:
		allRes = res
		for i in range(1,args.n):
			allRes += comm.recv(source=i, tag=0)
		
		# What to do once all data has been collected?
		resList = []
		for i in allRes:
			resList += i
	else:
		comm.send(res, dest=0, tag=0)

	# Timing
	if rank == 0:
		endTime = time.time()
		print("Ran for " + str(endTime - startTime) + " seconds")

