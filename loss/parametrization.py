import torch
from torch.nn import Module

from .mixin import DifferentialLossMixin
from utils import DifferentialMixin


class BaseParametrizationLoss(DifferentialLossMixin,
                            DifferentialMixin, Module):

    def __init__(self, reg_fold):
        super().__init__()

        self.reg_fold = reg_fold

        # variables used to compute distortion measures
        self.register_buffer('zero', torch.tensor(0.0))
        self.register_buffer('eye', torch.eye(2))
        self.register_buffer('eps',  torch.tensor(0.01))
        self.register_buffer('one',  torch.tensor(1.0))


    def forward(self, points_3D, source, mapped):
        # compute the First Fundamental Form(FFF) of the transformation mapped -> points_3D
        FFF, J_h = self.compute_jacobians(points_3D, source, mapped)

        # compute the distortion energy from FFF
        ppd = self.distortion(FFF)

        # add injectivity penalization
        pp_fold = self.compute_fold_reg(J_h)
        # compute average energy
        loss    = (ppd + self.reg_fold * pp_fold).mean()

        return loss


    def compute_jacobians(self, points_3D, source, mapped):
        J_h = self.gradient(mapped, source)
        J_f = self.gradient(points_3D, source)
        J_h_inv = J_h.inverse()
        J = J_f.matmul(J_h_inv)

        ### First Fundamental Form
        FFF = J.transpose(1, 2).matmul(J)

        return FFF, J_h


    def distortion(self, FFF):
        raise NotImplementedError()


class IsometricParamLoss(BaseParametrizationLoss):
    # use symmetric dirichlet as energy
    def distortion(self, FFF):
        return self.symm_dirichlet(FFF)

class ConformalParamLoss(BaseParametrizationLoss):
    # use conformal as energy
    def distortion(self, FFF):
        return self.conformal(FFF)
