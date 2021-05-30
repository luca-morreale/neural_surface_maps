
import torch

from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import MapDataset
from models import SurfaceMapModel
from models import InterMapModel
from loss import SSDLoss
from loss import IsometricMapLoss
from loss import CircleBoundaryLoss
from loss import SDFLoss


class InterSurfaceMap(LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net  = InterMapModel() # neural map between domains
        self.meta = SurfaceMapModel() # surface map (fixed) 2D -> 3D

        self.map_loss   = IsometricMapLoss() # isometric energy
        self.lands_loss = SSDLoss()
        self.bound_loss = CircleBoundaryLoss()
        self.sdf_loss   = SDFLoss()


    def train_dataloader(self):
        self.dataset = MapDataset(self.config.dataset)
        dataloader   = DataLoader(self.dataset, batch_size=None, shuffle=True,
                                num_workers=self.config.dataset.num_workers)

        return dataloader


    def configure_optimizers(self):
        LR        = 1.0e-4
        optimizer = RMSprop(self.net.parameters(), lr=LR)
        restart   = int(self.config.dataset.num_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=restart)
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):

        source    = batch['source_g'] # Nx2
        weights_g = batch['weights_g'] # weights source
        weights_f = batch['weights_f'] # weights target
        boundary  = batch['boundary_g'] # Bx2
        R         = batch['R'] # 2x2
        lands_g   = batch['lands_g'] # Lx2
        lands_f   = batch['lands_f'] # Lx2
        C_g       = batch['C_g'] # float (area normalization factor)
        C_f       = batch['C_f'] # float (area normalization factor)

        # activate gradient for autodiff
        source.requires_grad_(True)

        rot_src   = source.matmul(R.t()) # pre apply rotation
        rot_bnd   = boundary.matmul(R.t())
        rot_lands = lands_g.matmul(R.t())

        map_src   = self.net(rot_src) # forward source to target domain
        map_bnd   = self.net(rot_bnd) # forward boundary
        map_lands = self.net(rot_lands) # forward landmarks
        G         = self.meta(source, weights_g) * C_g # forward surface source
        F         = self.meta(map_src, weights_f) * C_f # forward surface target

        loss_map   = self.map_loss(F, map_src, source, G)
        loss_bnd   = self.bound_loss(map_bnd)
        loss_lands = self.lands_loss(map_lands, lands_f)
        loss_sdf   = self.sdf_loss(map_src)

        loss = loss_map + 1.0e4 * (loss_bnd+loss_sdf) + 1.0e6 * loss_lands


        return loss
