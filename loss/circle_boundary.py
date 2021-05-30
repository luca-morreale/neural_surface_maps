
import torch
from torch.nn import Module


class CircleBoundaryLoss(Module):

    def __init__(self):
        super().__init__()

        self.register_buffer('one', torch.tensor(1.0))


    def forward(self, mapped):
        # compute distance for points to the boundary of unit circle
        # ||x||_2^2 - 1.0
        loss = (mapped.pow(2).sum(-1) - self.one).pow(2).sum()
        return loss
