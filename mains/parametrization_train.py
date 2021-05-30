
import torch

from pytorch_lightning.core.lightning import LightningModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import ParametrizationDataset
from loss import IsometricParamLoss
from models import InterMapModel
from models import SurfaceMapModel
# from loss import ConformalParamLoss


class ParametrizationMap(LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net  = InterMapModel() # neural map between domains
        self.meta = SurfaceMapModel() # surface map (fixed) 2D -> 3D
        # self.loss_function = IsometricParamLoss(100.0) # opt isometric energy
        self.loss_function = ConformalParamLoss(100.0) # conformal energy


    def train_dataloader(self):
        self.dataset = ParametrizationDataset(self.config.dataset)
        dataloader   = DataLoader(self.dataset, batch_size=None, shuffle=True,
                                num_workers=self.config.dataset.num_workers)

        return dataloader


    def configure_optimizers(self):
        LR        = 1.0e-6
        optimizer = RMSprop(self.net.parameters(), lr=LR)
        restart   = int(self.config.dataset.num_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=restart)
        # return [optimizer], [scheduler]
        return optimizer


    def training_step(self, batch, batch_idx):
        source = batch['source']   # Nx2
        weights = batch['weights'] # weights

        # activate gradient for autodiff
        source.requires_grad_(True)

        # forward surface map (fixed)
        G   = self.meta(source, weights)
        # forward map for parametrization
        out = self.net(source)

        loss = self.loss_function(G, source, out)

        # add logging here if needed

        return loss
