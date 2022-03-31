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

# Simple CNN model
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
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
	parser.add_argument('-i', dest='i', type=int, help='Number of test images', default=7180)
	parser.add_argument('-g', dest='g', type=int, help='How many GPUs to use', default=0)
	parser.add_argument('-r', dest='r', type=int, help='Number of times to run', default=1)
	parser.add_argument('-o', dest='o', help='Output file, will be overwritten')
	args = parser.parse_args()

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
	train = data.DataLoader(dataset=train, batch_size=128, shuffle=True)
	test = DataClass(split='test', transform=data_transform, download=True)

	# Create larger dataset, if necessary
	while args.i > len(test):
		# Makes a list longer than needed, but doesn't matter
		test = test + test

	model = Net(in_channels=n_channels, num_classes=n_classes)

	# Train if no parameters provided
	if type(args.f) == type(None):
		model = model.cuda()
		#train = train.cuda()
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
				outputs = model(inputs.cuda())
				
				if task == 'multi-label, binary-class':
					targets = targets.to(torch.float32)
					loss = criterion(outputs, targets)
				else:
					targets = targets.squeeze().long()
					loss = criterion(outputs, targets.cuda())
				
				loss.backward()
				optimizer.step()
		torch.save(model, 'model.pt')
	else:
		model = torch.load(args.f)

	times = []
	for k in range(args.r):
		startTime = time.time()

		# Inference time
		res = []
		batchSize = 512
		testLoad = data.DataLoader(test, batch_size=batchSize, shuffle=False)

		with torch.no_grad():
			if args.g > 0:
				if torch.cuda.is_available():
					if torch.cuda.device_count() >= args.g:
						#numDevices = torch.cuda.device_count()
						gpuStr = 'cuda'
						devIds = []
						for i in range(args.g):
							#if i > 0:
							#	gpuStr += ','
							#gpuStr += str(i)
							devIds.append(i)
						device = torch.device(gpuStr)
					model.to(device)
					model = nn.DataParallel(model,device_ids = devIds)
					#testLoad = testLoad.to(device)
				else:
					print("ERROR: Could not use CUDA")
					quit()
			for inputs, targets in testLoad:
				res.append(model(inputs.cuda()).softmax(dim=-1))

		# What to do once all data has been collected?
		resList = []
		for i in res:
			resList += i

		# Timing
		endTime = time.time()
		times.append(endTime-startTime)

	f = open(args.o, "w")
	counter = 1
	for i in times:
		f.write(str(counter) + ',' + str(i) + '\n')
		counter += 1
	f.write('Average,' + str(sum(times)/len(times)) + '\n')
	f.close()
