import torch
from torch import nn

import einops.layers.torch


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(*[
            nn.Conv2d(3, 4, 3),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(4, 8, 3),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(8, 8, 3),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(8, 10, 3),
            nn.MaxPool2d(6, 6),
            einops.layers.torch.Rearrange('b c h w -> b (c h w)'),
            nn.Linear(10, 10),
            nn.LogSoftmax(dim=-1),
        ])
    
    def forward(self, x):
        return self.seq(x)
    