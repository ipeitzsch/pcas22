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
# from mpi4py import MPI


# Simple CNN model
class Net(nn.Module):
    def __init__(self, in_channels, num_classes, split_size=20, numGPU=4):
        super(Net, self).__init__()

        self.gpus = [0] * 6
        for i in range(len(self.gpus)):
            self.gpus[i] = i % numGPU
        self.gpus.sort()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU()).to(f'cuda:{self.gpus[0]}')

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(f'cuda:{self.gpus[1]}')

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()).to(f'cuda:{self.gpus[2]}')

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()).to(f'cuda:{self.gpus[3]}')

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(f'cuda:{self.gpus[4]}')

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)).to(f'cuda:{self.gpus[5]}')

        self.splitSize = split_size

    def forward(self, x):
        prevs = [None] * 6
        splits = iter(x.split(1, dim=0))
        sNext = next(splits)
        prevs[0] = sNext.to(f'cuda:{self.gpus[0]}')
        ret = []
        count = 1
        first = True
        for sNext in splits:
            count += 1
            if prevs[5] is not None:
                x = self.fc(prevs[5].view(prevs[5].size(0), -1))
                ret.append(x.softmax(dim=-1))
            if prevs[4] is not None:
                prevs[5] = self.layer5(prevs[4]).to(f'cuda:{self.gpus[5]}')
            if prevs[3] is not None:
                prevs[4] = self.layer4(prevs[3]).to(f'cuda:{self.gpus[4]}')
            if prevs[2] is not None:
                prevs[3] = self.layer3(prevs[2]).to(f'cuda:{self.gpus[3]}')
            if prevs[1] is not None:
                prevs[2] = self.layer2(prevs[1]).to(f'cuda:{self.gpus[2]}')
            if prevs[0] is not None:
                prevs[1] = self.layer1(prevs[0]).to(f'cuda:{self.gpus[1]}')
            prevs[0] = sNext.to(f'cuda:{self.gpus[0]}')
            
        while (prevs[5] is not None) or (first):
            if prevs[5] is not None:
                x = self.fc(prevs[5].view(prevs[5].size(0), -1))
                ret.append(x.softmax(dim=-1))
            if prevs[4] is not None:
                prevs[5] = self.layer5(prevs[4]).to(f'cuda:{self.gpus[5]}')
                first = False
            else:
                prevs[5] = None
            if prevs[3] is not None:
                prevs[4] = self.layer4(prevs[3]).to(f'cuda:{self.gpus[4]}')
            else:
                prevs[4] = None
            if prevs[2] is not None:
                prevs[3] = self.layer3(prevs[2]).to(f'cuda:{self.gpus[3]}')
            else:
                prevs[3] = None
            if prevs[1] is not None:
                prevs[2] = self.layer2(prevs[1]).to(f'cuda:{self.gpus[2]}')
            else:
                prevs[2] = None
            if prevs[0] is not None:
                prevs[1] = self.layer1(prevs[0]).to(f'cuda:{self.gpus[1]}')
            else:
                prevs[1] = None
            prevs[0] = None

        # print(f'count: {count}')
        r = torch.cat(ret)
        return r


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
    parser.add_argument('-g', dest='g', type=int, help='How many GPUs to use', default=1)
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
    testLoad = data.DataLoader(test, batch_size=512, shuffle=False)
    # Create larger dataset, if necessary
    while args.i > len(test):
        # Makes a list longer than needed, but doesn't matter
        test = test + test
    print(f'Num GPUS: {torch.cuda.device_count()}')
    model = Net(in_channels=n_channels, num_classes=n_classes, split_size=512, numGPU=torch.cuda.device_count())

    times = []
    for k in range(args.r):
        startTime = time.time()

        # Inference time
        res = []
        batchSize = 512


        with torch.no_grad():
            for inputs, targets in testLoad:
                res.append(model(inputs.cuda()))

        # What to do once all data has been collected?
        resList = []
        for i in res:
            resList += i

        # Timing
        endTime = time.time()
        times.append(endTime - startTime)

    f = open(args.o, "w")
    counter = 1
    for i in times:
        f.write(str(counter) + ',' + str(i) + '\n')
        counter += 1
    f.write('Average,' + str(sum(times) / len(times)) + '\n')
    f.close()

