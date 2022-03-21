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

# CUDA
from numba import jit

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

if __name__ == "__main__":
	# Arg parser
	parser = argparse.ArgumentParser(description='Run neural network using data parallel on multiple GPUs')
	parser.add_argument('-f', dest='f', help='File containing description of neural network')
	parser.add_argument('--num', dest='n', type=int, help='Number of nodes to use')
	parser.add_argument('-i', dest='i', type=int, help='Number of test images')
	parser.add_argument('-g', dest='g', action='store_true', help='Use GPUs')
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
	train = data.DataLoader(dataset=train, batch_size=128, shuffle=True)
	test = data.DataLoader(dataset=test, batch_size=256, shuffle=False)

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
		pVec = nn.utils.parameters_to_vector(model)
		print(type(pVec))
	else:
		pList = []
		f = csv.reader(open(args.f, 'r'))[0]
		for i in f:
			pList.append(float(i))
		pVec = np.array(pList)
		nn.utils.vector_to_parameters(pVec, model)
	
	# Start timer
	if rank == 0:
		startTime = time.time()

	# Now, distribute images
	data = []
	if rank == 0:
		for i in range(args.i):
			if i % args.n == 0:
				data.append(test[i%len(test)])
			else:
				req = comm.isend(test[i%len(test)], dest=i%args.n, tag=0)
				req.wait()
	else:
		numResp = int(args.i/args.n)
		if rank < args.i%args.n:
			numResp += 1
		for i in range(numResp):
			req = comm.irecv(source=0, tag=0)
			data.append(req.wait())

	# Inference time
	res = []
	testLoad = data.DataLoader(data, batch_size=256, shuffle=False)
	
	with torch.no_grad():
		for inputs, targets in testLoad:
			if args.g:
				res.append(jit(model(inputs)))
			else:
				res.append(model(inputs))
	
	# Recover all inferences at root
	count = 0
	if rank == 0:
		allRes = []
		for i in range(args.i):
			if % args.n != 0:
				req = comm.irecv(source=i%args.n, tag=0)
				allRes.append(req.wait())
			else:
				allRes.append(res[count])
				count += 1
		
		# What to do once all data has been collected?
	else:
		for i in res:
			req = comm.isend(i, dest=0, tag=0)
	
	# Timing
	if rank == 0:
		endTime = time.time()
		print("Ran for " + str(endTime - startTime) + " seconds")
