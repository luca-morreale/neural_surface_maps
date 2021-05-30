import torch
from torch.nn import Module

from .mixin import DifferentialLossMixin
from utils import DifferentialMixin


class BaseMapLoss(DifferentialLossMixin,
                    DifferentialMixin, Module):

    def __init__(self):
        super().__init__()

        self.reg_fold = 100

        # variables used to compute distortion measures
        self.register_buffer('zero', torch.tensor(0.0))
        self.register_buffer('eye', torch.eye(2))
        self.register_buffer('eps',  torch.tensor(0.01))
        self.register_buffer('one',  torch.tensor(1.0))


    def forward(self, points_3D_F, mapped, source, points_3D_G):
        # compute bool mask for points inside and outside the domain
        with torch.no_grad():
            out_mask = self.get_out_mask(mapped)
            in_mask  = ~out_mask

        # compute the First Fundamental Form(FFF) of the transformation points_3D_G -> points_3D_F
        FFF, J_h = self.compute_jacobians(points_3D_F, mapped, source, points_3D_G)

        # compute the distortion energy from FFF
        ppd = self.distortion(FFF)

        # add injectivity penalization
        pp_fold = self.compute_fold_reg(J_h)

        # assemble all the energy components
        pp_distortion = ppd + self.reg_fold * pp_fold
        # filter out points outside the domain
        pp_distortion[out_mask] = 0.0

        loss = pp_distortion.sum() / in_mask.long().sum()

        return loss

    def compute_jacobians(self, F, mapped, source, G):
        '''
            F : 3D points Nx3
            mapped : 2D points in the domain of F Nx2
            source : 2D points in the domain of G Nx2
            G : 3D points Nx3
        '''

        J_f = self.gradient(out=F, wrt=mapped)
        J_h = self.gradient(out=mapped, wrt=source)
        J_fh = J_f.matmul(J_h)

        J_g     = self.gradient(out=G, wrt=source)
        J_g_inv = self.invert_J(J_g)

        J = J_fh.matmul(J_g_inv)

        FFF = J.transpose(1,2).matmul(J)

        return FFF, J_h

    def get_out_mask(self, mapped):
        # points outside the unit disk are considered outside the domain
        norm_uv  = mapped.pow(2).sum(-1)
        out_mask = (norm_uv > 1.0).bool()

        return out_mask

    def distortion(self, FFF):
        raise NotImplementedError()


class IsometricMapLoss(BaseMapLoss):

    def distortion(self, FFF):
        return self.symm_dirichlet(FFF)

class ConformalMapLoss(BaseMapLoss):

    def distortion(self, FFF):
        return self.conformal(FFF)
