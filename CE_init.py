import argparse
import sys
import builtins
import os
import random
import shutil
import time
import warnings
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import CIFAR100
from cifar_noisy import CIFAR100_noisy
from sklearn import manifold
import numpy as np
from sklearn import manifold
from model import Model
from train import get_config


np.random.seed(0)
args = get_config()

# Añadir nuevos atributos si no existen
if not hasattr(args, 'self_sup_type'):
    args.self_sup_type = 'random'

if not hasattr(args, 'noise_type'):
    args.noise_type = 'instance'

if not hasattr(args, 'noise_rate'):
    args.noise_rate = 0.6
def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device

args.G = 50
args.max_iter = 1500
args.device = set_device()



class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        model = Model().cuda()
        model = nn.DataParallel(model)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path),strict=False)
        self.f = model.module.f
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return feature, out


if args.self_sup_type == 'simclr':
    print('initialization with simclr')
    ssl_model_path = 'pretrained/simclr.pth'
    model  = Net(num_class= args.num_classes, pretrained_path=ssl_model_path).cuda()
elif args.self_sup_type == 'ipirm':
    print('initialization with ipirm')
    ssl_model_path = 'pretrained/ipirm.pth'
    model = Net(num_class= args.num_classes, pretrained_path=ssl_model_path).cuda()
elif args.self_sup_type == 'random':
    print('random initialization')



train_loader = args.train_loader
test_loader = args.test_loader

model.cuda()

criterion = nn.CrossEntropyLoss().cuda()



best_acc = [0]
def validate(val_loader, model, criterion):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda()
            # compute output
            feature,logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
            acc = 100*float(correct)/float(total) 
    return acc



optimizer = optim.SGD(model.parameters(), lr=0.1)
alpha_plan = [0.1] * 50 + [0.01] * (args.epochs - 50)
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]


for epoch in range(args.epochs):
    model.train()
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    for i, (images, labels,true_labels,indexes) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        feature,output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc1 = validate(test_loader, model, criterion)
    if acc1>best_acc[0]:
        best_acc[0] = acc1
    print('current epoch',epoch)
    print('best acc',best_acc[0])
    print('last acc', acc1)












