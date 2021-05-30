
import torch

from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import CollectionDataset
from models import SurfaceMapModel
from models import InterMapModel
from loss import SSDLoss
from loss import IsometricCollectionLoss
from loss import CircleBoundaryLoss
from loss import SDFLoss


class CollectionMap(LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net_f = InterMapModel() # neural map between domains
        self.net_q = InterMapModel() # neural map between domains
        self.meta  = SurfaceMapModel() # surface map (fixed) 2D -> 3D

        self.map_loss   = IsometricCollectionLoss() # isometric energy
        self.lands_loss = SSDLoss()
        self.bound_loss = CircleBoundaryLoss()
        self.sdf_loss   = SDFLoss()


    def train_dataloader(self):
        self.dataset = CollectionDataset(self.config.dataset)
        dataloader   = DataLoader(self.dataset, batch_size=None, shuffle=True,
                                num_workers=self.config.dataset.num_workers)

        return dataloader


    def configure_optimizers(self):
        LR        = 1.0e-4
        optimizer = RMSprop(list(self.net_f.parameters()) +
                        list(self.net_q.parameters()), lr=LR)
        restart   = int(self.config.dataset.num_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=restart)
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):

        source    = batch['source_g'] # Nx2
        weights_g = batch['weights_g'] # weights
        weights_f = batch['weights_f'] # weights
        weights_q = batch['weights_q'] # weights
        boundary  = batch['boundary_g'] # Bx2
        R_f       = batch['R_f'] # 2x2
        R_q       = batch['R_q'] # 2x2
        lands_g   = batch['lands_g'] # Lx2
        lands_f   = batch['lands_f'] # Lx2
        lands_q   = batch['lands_q'] # Lx2
        C_g       = batch['C_g'] # float (area normalization factor)
        C_f       = batch['C_f'] # float (area normalization factor)
        C_q       = batch['C_q'] # float (area normalization factor)

        # activate gradient for autodiff
        source.requires_grad_(True)

        # pre apply rotation
        rot_src_f = source.matmul(R_f.t())
        rot_src_q = source.matmul(R_q.t())

        rot_bnd_f = boundary.matmul(R_f.t())
        rot_bnd_q = boundary.matmul(R_q.t())

        rot_lands_f = lands_g.matmul(R_f.t())
        rot_lands_q = lands_g.matmul(R_q.t())

        # forward source to target domain
        map_src_f = self.net_f(rot_src_f)
        map_src_q = self.net_q(rot_src_q)

        # forward boundary
        map_bnd_f   = self.net_f(rot_bnd_f)
        map_bnd_q   = self.net_q(rot_bnd_q)

        # forward landmarks
        map_lands_f = self.net_f(rot_lands_f)
        map_lands_q = self.net_q(rot_lands_q)

        # forward surface
        G = self.meta(source, weights_g)    * C_g
        F = self.meta(map_src_f, weights_f) * C_f
        Q = self.meta(map_src_q, weights_q) * C_q


        loss_map = self.map_loss(F, map_src_f, Q, map_src_q, source, G)

        loss_bnd   = self.bound_loss(map_bnd_f) + \
                        self.bound_loss(map_bnd_q)

        loss_lands = self.lands_loss(map_lands_f, lands_f) + \
                        self.lands_loss(map_lands_q, lands_q)

        loss_sdf   = self.sdf_loss(map_src_f) + self.sdf_loss(map_src_q)

        loss = loss_map + 1.0e6 * (loss_bnd+loss_sdf) + 1.0e8 * loss_lands

        return loss
