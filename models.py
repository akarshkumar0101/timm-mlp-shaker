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
    
    

# def perform_stats(self, net, loader=None, show_stats=True, n_batches=-1, tqdm=tqdm, device='cpu'):
#         if loader is None:
#             loader = self.loader_test
#         n_correct, total = 0, 0
#         loss_total = 0
#         n_examples = 0
#         loop = enumerate(loader)
#         if tqdm is not None:
#             loop = tqdm(loop, leave=False, total=max(n_batches, len(loader)))
#         for batch_idx, (X_batch, Y_batch) in loop:
#             X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
#             if batch_idx == n_batches:
#                 break
#             Y_batch_pred = net(X_batch)
#             n_correct += (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()
#             loss = self.loss_func(Y_batch_pred, Y_batch).item()
#             loss_total += loss * len(X_batch)
#             n_examples += len(X_batch)
#             total += len(Y_batch)
#         loss_total /= n_examples
#         accuracy = n_correct/total*100.
#         if show_stats:
#             print(f'Average Loss: {loss_total:.03f}, Accuracy: {accuracy:.03f}%')
#         return {'loss': loss_total, 'accuracy': accuracy}