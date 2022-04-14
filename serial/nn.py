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

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU())
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=3, padding=1),
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
	parser = argparse.ArgumentParser(description='Run serial neural network on one GPU')
	parser.add_argument('-f', dest='f', help='File containing description of neural network')
	parser.add_argument('--num', dest='n', type=int, help='Number of nodes to use', default=1)
	parser.add_argument('-i', dest='i', type=int, help='Number of test images', default=7180)
	parser.add_argument('-g', dest='g', action='store_true', help='Use GPUs')
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
	test = DataClass(split='test', transform=data_transform, download=True)

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
	startTime = time.time()

	# Inference time
	res = []
	batchSize = 256
	testLoad = data.DataLoader(test, batch_size=batchSize, shuffle=False)

	model.eval()
	y_true = torch.tensor([])
	y_score = torch.tensor([])

	with torch.no_grad():
		for inputs, targets in testLoad:
			res.append(model(inputs).softmax(dim=-1))

			y_true = torch.cat((y_true, targets), 0)
			y_score = torch.cat((y_score, outputs), 0)

		y_true = y_true.numpy()
		y_score = y_score.detach().numpy()

		evaluator = Evaluator(data_flag, 'test')
		metrics = evaluator.evaluate(y_score)

		print('%s  auc: %.3f  acc:%.3f' % ('test', *metrics))

	# Timing
	endTime = time.time()
	print("Ran for " + str(endTime - startTime) + " seconds")

	file = open("serial-times.csv", "a")
	file.write("{}\n".format(str(endTime - startTime)))