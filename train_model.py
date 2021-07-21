import sys
print(sys.argv)

import scipy
import scipy.stats
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import copy
import os
import shutil
import math

import torchinfo
from torchinfo import summary
# import torch.utils.tensorboard as tb

np.random.seed(0)
torch.manual_seed(10);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import util
import imagenet
from imagenet import ImageNetN


from train_simple import train_net, evaluate_net


bs_train, bs_test = 64, 128
# bs_train, bs_test = 512, 1024
work_dir = os.environ['WORK']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

ds_train = torchvision.datasets.CIFAR10(f'{work_dir}/datasets/cifar10', train=True, transform=transform)
ds_test = torchvision.datasets.CIFAR10(f'{work_dir}/datasets/cifar10', train=False, transform=transform)
# ds_train = torchvision.datasets.ImageNet(f'{work_dir}/datasets/imagenet', split='train', transform=transform)
dl_train = DataLoader(ds_train, batch_size=bs_train, shuffle=True, num_workers=16)
dl_test = DataLoader(ds_test, batch_size=bs_test, shuffle=True, num_workers=16)
n_classes = len(ds_train.classes)
bs, c, h, w = input_shape = next(iter(dl_train))[0].shape


import mlp_shaker_flat
import mlp_mixer_pytorch
import torchinfo

if sys.argv[1]=='mix':
    net = mlp_mixer_pytorch.MLPMixer(image_size=224, channels=3, patch_size=16, 
                                     dim=16*16*3, depth=3, num_classes=n_classes, 
                                     expansion_factor=4.0, dropout=0.2)
elif sys.argv[1]=='my_mix':
    net = mlp_shaker_flat.ViMixer(image_size=224, channels=3, patch_size=16, 
                                  dim=16*16*3, depth=3, num_classes=n_classes, 
                                  expansion_factor=4.0, dropout=0.2)
elif sys.argv[1]=='shake':
    net = mlp_shaker_flat.ViShaker(image_size=224, channels=3, patch_size=16, 
                                  dim=16*16*3, depth=3, num_classes=n_classes, 
                                  expansion_factor=4.0, dropout=0.2)
    
dev = int(sys.argv[2])
device = f'cuda:{dev}'
    
net = net.cuda(dev)
summary(net, input_size=input_shape, device=device)

evaluate_net(net, dl=dl_test, tqdm=tqdm, device=device, verbose=True)
losses_train, losses_test = train_net(net, dl=dl_train, dl_test=dl_test, n_epochs=10, device=device, lr=1e-2, tqdm=tqdm)
evaluate_net(net, dl=dl_test, tqdm=tqdm, device=device, verbose=True)