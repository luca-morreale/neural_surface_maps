import torch

from losses.mixin import DifferentialLossMixin

class AnalyticalMixin:

    def compute_jacobians(self, F, mapped, source, G):
        '''
            F : 3D points Nx3
            mapped : 2D points in the domain of F Nx2
            source : 2D points in the domain of G Nx2
            G : 3D points Nx3
        '''

        J_f  = self.gradient(out=F, wrt=mapped)
        J_h  = self.gradient(out=mapped, wrt=source)
        J_fh = J_f.matmul(J_h)

        J_g     = self.gradient(out=G, wrt=source)
        J_g_inv = self.invert_J(J_g)

        J = J_fh.matmul(J_g_inv)

        FFF = J.transpose(1,2).matmul(J)

        return FFF, J_h

    def get_out_mask(self, mapped, source):
        v1 = source[:, 0]
        v2 = source[:, 1]
        v3 = source[:, 2]
        point_mask = self.point_in_triangle(mapped.view(-1, 1, 2),
                                v1[None, :], v2[None, :], v3[None, :])

        point_mask = point_mask.sum(dim=-1).bool() # point is contained in at least a triangle

        return ~point_mask


    def compute_point_area(self, FFF):
        E, F, _, G = FFF.view(-1, 4).transpose(1,0)

        ### Get per point local squared area.
        A2 = torch.max(E * G - F.pow(2), self.zero)  # (B, P, spp)
        A = A2#.sqrt()  # (B, P, spp)
        return A

