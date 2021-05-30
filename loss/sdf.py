
import torch
from torch.nn import Module


class SDFLoss(Module):
    '''
        Signed Distance Function

        Points inside the domain have a norm < 1.0
        Points outside the domain have a norm > 1.0

        norm = ||x_i||_2^2
        norm [norm < 1.0] = 0
        mean(norm)
    '''

    def __init__(self):
        super().__init__()
        self.register_buffer('one',  torch.tensor(1.0))
        self.register_buffer('zero', torch.tensor(0.0))

    def forward(self, points):
        norm = points.pow(2).sum(-1)
        sdf  = torch.max(norm - self.one, self.zero) # set norm for points inside = 0
        return sdf.mean()
