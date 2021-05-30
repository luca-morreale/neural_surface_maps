
import torch
from torch.nn import Module

from torch.nn import Softplus

from utils import get_init_fun
from utils import create_sequential_metalinear_layer


class InterMapModel(Module):

    def __init__(self):
        super().__init__()

        input_size  = 2
        output_size = 2
        act_fun     = Softplus

        # network structure
        layers = [input_size,128,128,128,128,output_size]
        # create a sequential module
        self.mlp = create_sequential_metalinear_layer(layers, act_fun, last_act=False)

        ## initialize weights
        init_fun = get_init_fun()
        self.mlp.apply(init_fun)


    def forward(self, x):

        x = self.mlp(x)
        return x
