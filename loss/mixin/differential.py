import torch

class DifferentialLossMixin:


    def compute_fold_reg(self, J):
        '''
            Penalize negative eigenvalues to enforce injectivity of the map
        '''
        J_det = J.det()
        J_det_sign = torch.sign(J_det)
        pp_fold = torch.max(-J_det_sign * torch.exp(-J_det), self.zero)

        return pp_fold


    def conformal(self, FFF):
        '''
            Compute conformal energy from First Fundamental Form (FFF)
        '''
        E = FFF[:, 0,0]
        G = FFF[:, 1,1]

        ### conformal: || _lambda * M - I ||
        lambd = (E + G) / FFF.pow(2).sum(-1).sum(-1)
        ppd   = (lambd.view(-1, 1, 1) * FFF - self.eye).pow(2).sum(-1).sum(-1)
        return ppd


    def symm_dirichlet(self, FFF):
        '''
            Compute symmetric dirichlet energy from First Fundamental Form (FFF)
        '''
        FFF_inv = (FFF + self.eps).inverse()

        E = FFF[:, 0,0] - 1.0
        G = FFF[:, 1,1] - 1.0
        E_inv = FFF_inv[:, 0,0] - 1.0
        G_inv = FFF_inv[:, 1,1] - 1.0

        ### symm dirichlet: trace(J^T J) + trace(J_inv^T J_inv )
        dirichlet = E + G
        inv_dirichlet = E_inv + G_inv
        ppd = dirichlet + inv_dirichlet
        return ppd


    def arap(self, FFF):
        '''
            Compute ARAP energy from First Fundamental Form (FFF)
        '''
        ### arap: eigv(J) - I
        ppd = (FFF - self.eye).pow(2).sum(-1).sum(-1)
        return ppd
