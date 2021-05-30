import torch
from torch.nn import Module

from .mixin import DifferentialLossMixin
from utils import DifferentialMixin

class BaseCollectionLoss(DifferentialLossMixin,
                    DifferentialMixin, Module):

    def __init__(self):
        super().__init__()

        self.reg_fold = 100

        # variables used to compute distortion measures
        self.register_buffer('zero', torch.tensor(0.0))
        self.register_buffer('eye',  torch.eye(2))
        self.register_buffer('eps',  torch.tensor(0.01))
        self.register_buffer('one',  torch.tensor(1.0))


    def forward(self, F, map_src_f, Q, map_src_q, source, G):
        # compute bool mask for points inside and outside the domain for each map
        with torch.no_grad():
            out_mask_f = self.get_out_mask(map_src_f)
            out_mask_q = self.get_out_mask(map_src_q)
            out_mask = out_mask_f + out_mask_q
            in_mask  = ~out_mask

        ############## Per Point LOSS ##############
        # compute the energy for each map
        pp_distortion = self.cycle_distortion(F, Q, source, G)
        # add injectivity penalization
        pp_distortion += self.regularizer_folds(map_src_f, source) + \
                            self.regularizer_folds(map_src_q, source)

        ############# FILTERING ##############
        pp_distortion[out_mask] = 0.0 # filter out points outside the domain

        ############## LOSS ##############
        loss_distortion = pp_distortion.sum() / in_mask.sum().float()

        return loss_distortion


    def cycle_distortion(self, F, Q, source, G):
        '''
            Compute the distortion for each possible map pair. all going through the hub (G)
            G -> F
            G -> Q
            F -> Q (consistency part)
        '''

        J_fh = self.gradient(out=F, wrt=source)
        J_qh = self.gradient(out=Q, wrt=source)

        J_g     = self.gradient(out=G, wrt=source)
        J_g_inv = self.invert_J(J_g)

        J_fh_inv = self.invert_J(J_fh)

        J_fhg = J_fh.matmul(J_g_inv)
        J_qhg = J_qh.matmul(J_g_inv)
        J_qhf = J_qh.matmul(J_fh_inv)

        FFF_fhg = J_fhg.transpose(1,2).matmul(J_fhg)
        FFF_qhg = J_qhg.transpose(1,2).matmul(J_qhg)
        FFF_qhf = J_qhf.transpose(1,2).matmul(J_qhf)

        ppd_fhg = self.distortion(FFF_fhg)
        ppd_qhg = self.distortion(FFF_qhg)
        ppd_qhf = self.distortion(FFF_qhf)

        return ppd_fhg + ppd_qhg + ppd_qhf


    def regularizer_folds(self, mapped_points, source):
        J = self.gradient(out=mapped_points, wrt=source)
        return self.compute_fold_reg(J)

    def get_out_mask(self, mapped):

        norm_uv  = mapped.pow(2).sum(-1)
        out_mask = (norm_uv > 1.0).bool()

        return out_mask


class IsometricCollectionLoss(BaseCollectionLoss):

    def distortion(self, FFF):
        return self.symm_dirichlet(FFF)

class ConformalCollectionLoss(BaseCollectionLoss):

    def distortion(self, FFF):
        return self.conformal(FFF)
