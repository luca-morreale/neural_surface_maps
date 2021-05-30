
import torch
from torch.nn import Module

class SSDLoss(Module):
    '''
        Sum of Squared Differences

        SSD = sum( x_i - y_i)
    '''

    def __init__(self):
        super().__init__()


    def forward(self, pred, gt):
        B    = gt.size(0)
        loss = (gt - pred).pow(2).view(B, -1).sum()

        return loss

