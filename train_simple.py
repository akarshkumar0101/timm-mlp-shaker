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

import copy
import os
import shutil
import math

import torchinfo
from torchinfo import summary
# import torch.utils.tensorboard as tb
import wandb

np.random.seed(0)
torch.manual_seed(10);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import util
import imagenet
from imagenet import ImageNetN

loss_fn = nn.CrossEntropyLoss()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_net(net, dl=None, tqdm=None, device=None, verbose=True):
    net = net.to(device)
    lossm, accm = AverageMeter(), AverageMeter()
    loop = enumerate(dl)
    if tqdm is not None:
        loop = tqdm(loop, total=len(dl))
    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        with torch.no_grad():
            Y_batch_pred = net(X_batch)
        loss = loss_fn(Y_batch_pred, Y_batch)
        acc = (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()/len(X_batch)
        lossm.update(loss.item(), n=len(X_batch))
        accm.update(acc, n=len(X_batch))
        
        if tqdm is not None: loop.set_postfix({'loss':loss.item()})
    if verbose:
        print(f'Average Loss: {lossm.avg:.03f}, Accuracy: {accm.avg:.03f}%')
    return {'loss': lossm.avg, 'accuracy': accm.avg}
    
def train_net(net, dl, dl_test=None, 
              n_epochs=10, lr=1e-2, device=None, 
              tqdm=None, verbose=True):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    wandb.watch(net)
    
    losses_batches = np.zeros((n_epochs, len(dl)))
    losses_train = np.zeros((n_epochs,))
    losses_test = np.zeros((n_epochs,))
    loop1 = range(n_epochs)
    
    if tqdm is not None:
        loop1 = tqdm(loop1, total=n_epochs)
        
    if dl_test is not None:
        data = evaluate_net(net, dl_test, tqdm=tqdm, device=device, verbose=verbose)
        wandb.log({'test_loss': data['loss'], 'test_accuracy': data['accuracy']})
    
    for epoch_idx in loop1:
        lossm, accm = AverageMeter(), AverageMeter()
        loop2 = enumerate(dl)
        if tqdm is not None:
            loop2 = tqdm(loop2, leave=False, total=len(dl))
        for batch_idx, (X_batch, Y_batch) in loop2:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_batch_pred = net(X_batch)

            loss = loss_fn(Y_batch_pred, Y_batch)
            acc = (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()/len(X_batch)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            lossm.update(loss.item(), n=len(X_batch))
            accm.update(acc, n=len(X_batch))
            wandb.log({'batch_loss': loss.item(), 'batch_accuracy': acc})
#             losses_train[epoch_idx, batch_idx] = loss.item()
            if tqdm is not None: loop1.set_postfix({'loss':loss.item()})
        wandb.log({'train_loss': lossm.avg, 'train_accuracy': accm.avg})
        if dl_test is not None:
            data = evaluate_net(net, dl_test, tqdm=tqdm, device=device, verbose=verbose)
            wandb.log({'test_loss': data['loss'], 'test_accuracy': data['accuracy']})
#     return losses_train, losses_test
