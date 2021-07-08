print('hey')

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
import torch.utils.tensorboard as tb

np.random.seed(0)
torch.manual_seed(10);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import util







h, w = 32, 32
nph, npw = 8, 8
bs_train, bs_test = 500, 1000
bs_train, bs_test = 10, 20

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
#     torchvision.transforms.Resize((h, w), ), 
#     Rearrange('c (nph psh) (npw psw) -> (nph npw) (psh psw c)', nph=8, npw=8),
])
ds_train = torchvision.datasets.CIFAR10('~/datasets/cifar10/', train=True, download=True, transform=transform)
ds_test = torchvision.datasets.CIFAR10('~/datasets/cifar10/', train=False, download=True, transform=transform)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs_train, shuffle=True,)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=bs_test, shuffle=True,)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



loss_fn = nn.NLLLoss()
def evaluate_model(net, dl=None, n_batches=-1, tqdm=tqdm, device=None, verbose=False):
    net = net.to(device)
    n_correct, total = 0, 0
    loss_total = 0
    n_examples = 0
    loop = enumerate(dl)
    if tqdm is not None:
        loop = tqdm(loop, leave=False, total=max(n_batches, len(dl)))
    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        if batch_idx == n_batches:
            break
        with torch.no_grad():
            Y_batch_pred = net(X_batch)
        n_correct += (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()
        loss = loss_fn(Y_batch_pred, Y_batch).item()
        loss_total += loss * len(X_batch)
        n_examples += len(X_batch)
        total += len(Y_batch)
    loss_total /= n_examples
    accuracy = n_correct/total*100.
    if verbose:
        print(f'Average Loss: {loss_total:.03f}, Accuracy: {accuracy:.03f}%')
    return {'loss': loss_total, 'accuracy': accuracy}
    
def train_classification_model(net, dl, n_epochs=10, device=None):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, )
    
    for epoch_idx in tqdm(range(n_epochs)):
        print(f'Starting epoch {epoch_idx}')
        loop = enumerate(dl)
        if tqdm is not None:
            loop = tqdm(loop, leave=False, total=len(dl))
        for batch_idx, (X_batch, Y_batch) in loop:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_batch_pred = net(X_batch)

            loss = loss_fn(Y_batch_pred, Y_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()
        evaluate_model(net, dl_test, device='cuda', verbose=True)
    




import mlp_shaker_flat
import mlp_mixer_pytorch
import torchinfo



#net = mlp_mixer_pytorch.MLPMixer(image_size=32, channels=3, patch_size=4, dim=48, depth=10, num_classes=10, expansion_factor=4, dropout=0.0)
#net(torch.randn(100, 3, 32, 32)).shape
#
#print('MLP MIXER')
#evaluate_model(net, dl=dl_test, verbose=True, device=device)
#train_classification_model(net, dl=dl_train, n_epochs=5, device=device)
#evaluate_model(net, dl=dl_test, verbose=True, device=device)


print('starting')
net = mlp_shaker_flat.ViMixer(image_size=32, channels=3, patch_size=4, dim=48, depth=10, num_classes=10, expansion_factor=4, dropout=0.0)
net(torch.randn(100, 3, 32, 32)).shape

evaluate_model(net, dl=dl_test, verbose=True, device=device)
train_classification_model(net, dl=dl_train, n_epochs=5, device=device)
evaluate_model(net, dl=dl_test, verbose=True, device=device)
print('ending')


#for depth in [10, 20, 40]:
#    net = mlp_shaker_flat.ViShaker(image_size=32, channels=3, patch_size=4, dim=48, depth=depth, num_classes=10, expansion_factor=4, dropout=0.0)
#    net(torch.randn(100, 3, 32, 32)).shape
#
#    evaluate_model(net, dl=dl_test, verbose=True, device=device)
#    train_classification_model(net, dl=dl_train, n_epochs=5, device=device)
#    evaluate_model(net, dl=dl_test, verbose=True, device=device)
