import torch
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import glob
import os
import torch
import torch.nn as nn
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def to_edge_index(graph, device):
    graph = graph.clone().detach().squeeze(0)
    if graph.dim() == 2 and graph.shape[0] == 2:
        return graph.to(torch.long).to(device)
    graph = graph.to(torch.float32).to(device)
    rows, cols = torch.nonzero(graph, as_tuple=True)
    return torch.stack([rows, cols]).to(device)

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_start_lr, end_lr, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.end_lr = end_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # linear warmup
        if self.last_epoch < self.warmup_epochs:
            lr = (self.base_lrs[0] - self.warmup_start_lr) * float(self.last_epoch) / float(self.warmup_epochs) + self.warmup_start_lr
            return [lr]

        # cosine annealing decay
        progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
        cosine_lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        lr = max(0.0, cosine_lr * (self.base_lrs[0] - self.end_lr) + self.end_lr)
        return [lr]
    def __str__(self):
        return (
            f"WarmupCosineLR(\n"
            f"  Current Epoch: {self.last_epoch}\n"
            f"  Warmup Start LR: {self.warmup_start_lr}\n"
            f"  End LR: {self.end_lr}\n"
            f"  Warmup Epochs: {self.warmup_epochs}\n"
            f"  Total Epochs: {self.total_epochs}\n"
            f")"
        )
