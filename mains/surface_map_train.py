
import torch

from pytorch_lightning.core.lightning import LightningModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import SurfaceMapDataset
from loss import SSDLoss
from models import SurfaceMapModel
from utils import DifferentialMixin


class SurfaceMap(DifferentialMixin, LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net = SurfaceMapModel() # map
        self.loss_function = SSDLoss() # loss



    def train_dataloader(self):
        self.dataset = SurfaceMapDataset(self.config.dataset)
        dataloader   = DataLoader(self.dataset, batch_size=None, shuffle=True,
                                num_workers=self.config.dataset.num_workers)

        return dataloader


    def configure_optimizers(self):
        LR        = 1.0e-4
        optimizer = RMSprop(self.net.parameters(), lr=LR, momentum=0.9)
        restart   = int(self.config.dataset.num_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=restart)
        return [optimizer], [scheduler]



    def training_step(self, batch, batch_idx):

        source  = batch['source']  # Nx2
        gt      = batch['gt']      # Nx3
        normals = batch['normals'] # Nx3

        # activate gradient so can compute the normals through differentiation
        source.requires_grad_(True)

        # forward network
        out          = self.net(source)
        # estimate normals through autodiff
        pred_normals = self.compute_normals(out=out, wrt=source)

        # loss
        loss_dist    = self.loss_function(out, gt)
        loss_normals = self.loss_function(pred_normals, normals)

        loss = loss_dist + 0.01 * loss_normals

        # add here logging if needed

        return loss
