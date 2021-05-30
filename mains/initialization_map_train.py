
import torch

from pytorch_lightning.core.lightning import LightningModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import InitalizationDataset
from loss import SSDLoss
from models import InterMapModel


class Initialization(LightningModule):
    # optimize map for identity map

    def __init__(self):
        super().__init__()

        self.net = InterMapModel()
        self.loss_function = SSDLoss()


    def train_dataloader(self):
        self.dataset = InitalizationDataset()
        dataloader   = DataLoader(self.dataset, batch_size=None, shuffle=True,
                                num_workers=4)
        return dataloader


    def configure_optimizers(self):
        LR        = 1.0e-5
        optimizer = RMSprop(self.net.parameters(), lr=LR, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=30000)
        return [optimizer], [scheduler]



    def training_step(self, batch, batch_idx):

        out  = self.net(batch)
        loss = self.loss_function(out, batch)

        return loss
