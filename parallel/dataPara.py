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
import threading
import copy

import medmnist
from medmnist import INFO, Evaluator

# Simple CNN model
class Net(nn.Module):
    def __init__(self, in_channels, num_classes, gpuNum):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()).to(gpuNum)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(gpuNum)

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU()).to(gpuNum)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU()).to(gpuNum)
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(gpuNum)

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)).to(gpuNum)

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

def runTestNetwork(model, dev, testData, batchSize, results, start):
	testLoad = data.DataLoader(testData, batch_size=batchSize, shuffle=False)
	count = 0
	with torch.no_grad():
		for inputs, targets in testLoad:
			inputs = inputs.to(dev)
			results[start+count] = model(inputs).softmax(dim=-1)
			count += 1

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
	train = data.DataLoader(dataset=train, batch_size=128, shuffle=True, num_workers=4)
	test = DataClass(split='test', transform=data_transform, download=True)

	# Convert to list for easier division
	temp = []
	for i in test:
		temp.append(i)
	test = temp

	# Create larger dataset, if necessary
	while args.i > len(test):
		# Makes a list longer than needed, but doesn't matter
		test = test + test

	"""
	model = Net(in_channels=n_channels, num_classes=n_classes, numGPU=torch.cuda.device_count())

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
	"""

	times = []
	with torch.no_grad():
		if args.g > 0:
			devList = []
			modelList = []
			if torch.cuda.is_available() and torch.cuda.device_count() == args.g:
				for i in range(args.g):
					tempStr = "cuda:"+str(i)
					devList.append(torch.device(tempStr))
					modelList.append(Net(in_channels=n_channels, num_classes=n_classes, gpuNum=tempStr))
			else:
				print("ERROR: Could not use CUDA")
				quit()
	
	with torch.no_grad():
		for k in range(args.r):
			startTime = time.time()

			# Inference time
			res = []
			batchSize = 2048
			testList = []
			startList = []
			stepSize = int(len(test)/args.g)
			extra = len(test) % args.g
			for i in range(args.g):
				start = i*stepSize
				if i > extra:
					start += extra
					end = start + stepSize
				else:
					start += i
					end = start + stepSize + 1
				startList.append(int(start))
				testList.append(test[int(start):int(end-1)])

			res = [0]*len(test)
			threads = []
			for i in range(args.g):
				threads.append(threading.Thread(target=runTestNetwork, args=(modelList[i], devList[i], testList[i], batchSize, res, int(startList[i]))))
				threads[-1].start()
			for i in threads:
				i.join()

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
