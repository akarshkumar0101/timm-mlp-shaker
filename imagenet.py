import numpy as np
import torch
import torchvision

class ImageNetN(torch.utils.data.Dataset):
    """ImageNet N classes dataset"""

    def __init__(self, root_dir, n_classes=1000, split='train', transform=None, imgnetn=None):
        self.root_dir = root_dir
        self.n_classes = n_classes
        self.split = split
        self.transform = transform
        self.ds = torchvision.datasets.ImageNet(root_dir, split=split, transform=transform)
        
        if imgnetn is not None:
            self.class_idxs = imgnetn.class_idxs
        elif len(self.ds.classes)==n_classes:
            self.class_idxs = np.arange(n_classes)
        else:
            self.class_idxs = np.random.permutation(len(self.ds.classes))[:n_classes]
            
        self.classes = np.array([i[0] for i in self.ds.classes])[self.class_idxs]
        idx_new2old = self.class_idxs
        idx_old2new = {e: idx for idx, e in enumerate(self.class_idxs)}
        self.imgs = [(x, idx_old2new[y]) for x, y in self.ds.imgs if y in self.class_idxs]
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        x, y = self.imgs[idx]
        x = self.ds.loader(x)
        if self.transform:
            x = self.transform(x)
        return x, y

