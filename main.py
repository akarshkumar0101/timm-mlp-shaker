import torch

print('device count:')
print(torch.cuda.device_count())

import numpy as np
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

import copy
import os
import shutil
import math

from torchinfo import summary
from functools import partial
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

np.random.seed(0)
torch.manual_seed(10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import wandb
import train_simple

import util

bs_train, bs_test = 512, 1024
work_dir = os.environ['WORK']
transform = transforms.Compose([
#     transforms.Resize((224, 224)),
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
# import mlp_mixer_pytorch
import torchinfo

# net = mlp_shaker_flat.ViShaker(image_size=32, channels=3, patch_size=4, 
#                               dim=4*4*3, depth=15, num_classes=n_classes, 
#                               expansion_factor=4.0, dropout=0.0)
# net = mlp_shaker_flat.ViMixer(image_size=32, channels=3, patch_size=4, 
#                               dim=4*4*3, depth=15, num_classes=n_classes, 
#                               expansion_factor=4.0, dropout=0.0)


device = 'cuda:3'
models = {'mixer': mlp_shaker_flat.ViMixer, 'shaker': mlp_shaker_flat.ViShaker}
for model_name, model in tqdm(models.items(), leave=False):
    for depth in tqdm([1, 4, 16, 32], leave=False):
        torch.manual_seed(0)
        np.random.seed(0)
        net = model(image_size=32, channels=3, patch_size=4, 
                    dim=4*4*3, depth=depth, num_classes=n_classes, 
                    expansion_factor=4.0, dropout=0.0)
        wandb.init(name=f'{model_name}_{depth:03d}')
        n_params = util.count_params(net)
        print(f'Training: {model_name} with depth: {depth} # params: {n_params}')
        
        lr = 1e-2
        config = {}
        config['model'] = copy.deepcopy(model_name)
        config['depth'] = copy.deepcopy(depth)
        config['lr'] = copy.deepcopy(lr)
        config['n_params'] = n_params

        train_simple.train_net(net, dl=dl_train, dl_test=dl_test, n_epochs=10, lr=lr, 
                               device=device, tqdm=partial(tqdm, leave=False))
        
        torch.save((config, net.cpu().state_dict()), f'data/{model_name}_{depth}.pth')
        wandb.finish()
