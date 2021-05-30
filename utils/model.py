
import math
import torch
import torchmeta

from torch.nn import Conv1d, Conv2d, Linear, Sequential
from torch.nn import ReLU, Tanh, SELU, ELU, Identity, Softplus, LeakyReLU
from torch.nn import Dropout
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import LayerNorm, LocalResponseNorm
from torch.nn import InstanceNorm1d as InstanceNorm

from torchmeta.modules import MetaLinear
from torchmeta.modules import MetaSequential



def create_sequential_linear_layer(layers_size, act_fun, last_act=True):
    def create_layer(in_feat, out_feat): # define how to create basic layer
        return Linear(in_feat, out_feat)

    layers  = [(in_feat, out_feat) for in_feat, out_feat in zip(layers_size[:-1], layers_size[1:])]
    return create_sequential(create_layer, layers, act_fun, last_act=last_act)

def create_sequential_metalinear_layer(layers_size, act_fun, last_act=True):
    def create_layer(in_feat, out_feat): # define how to create basic layer
        return MetaLinear(in_feat, out_feat)

    layers  = [(in_feat, out_feat) for in_feat, out_feat in zip(layers_size[:-1], layers_size[1:])]
    modules = create_sequential(create_layer, layers, act_fun, last_act=last_act)
    return MetaSequential(*modules) # convert Sequental to MetaSequential



def create_sequential(LayerClass, layers, act_fun, last_act=True):

    #################################
    # Create Sequential dynamically #
    #################################
    modules = []

    ## Generate a Sequential model based on the number of units specified in `model_structure`
    for (in_feat, out_feat) in layers:
        modules.append(LayerClass(in_feat, out_feat))
        modules.append(act_fun())

    if not last_act:
        modules = modules[:-1]

    return Sequential(*modules)



def get_init_fun():

    def initialize(m):
        init = torch.nn.init.xavier_normal_
        if type(m) == Conv1d or type(m) == Conv2d or type(m) == Linear or 'Linear' in type(m).__name__:
            init(m.weight)

    return initialize

