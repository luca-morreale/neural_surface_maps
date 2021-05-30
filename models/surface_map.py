
import torch
import torchmeta
from torchmeta.modules import MetaModule
from torchmeta.modules import MetaSequential
from torchmeta.modules import MetaLinear

from torch.nn import Softplus

from utils import get_init_fun
from utils import create_sequential_metalinear_layer


class SurfaceMapModel(MetaModule):

    def __init__(self):
        super().__init__()

        input_size  = 2
        output_size = 3
        act_fun     = Softplus

        modules = []

        # first layer input -> hiddent
        modules.append(MetaLinear(input_size, 256))
        modules.append(act_fun())

        # seq of residual blocks
        for layer in [256]*10:
            block = ResBlock(layer, act_fun)
            modules.append(block)

        # output layer
        modules.append(MetaLinear(256, output_size))

        self.mlp = MetaSequential(*modules)

        ## initialize weights
        init_fun = get_init_fun()
        self.mlp.apply(init_fun)


    def forward(self, x, params=None):

        x = self.mlp(x, params=self.get_subdict(params, 'mlp'))
        return x



class ResBlock(MetaModule):

    def __init__(self, in_features, act_fun):
        super().__init__()

        layer = create_sequential_metalinear_layer([in_features, in_features, in_features], act_fun)

        self.residual = MetaSequential(*layer[:-1])
        self.post_act = layer[-1]


    def forward(self, x, params=None):

        out = self.residual(x, params=self.get_subdict(params, 'residual'))
        out = self.post_act(out + x)
        return out
