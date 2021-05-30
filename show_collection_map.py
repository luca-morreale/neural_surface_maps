
import torch

from models import SurfaceMapModel
from models import InterMapModel

from utils import show_mesh


SURFACE_PATH_Q  = '/SET/HERE/YOUR/PATH'
SURFACE_PATH_F  = '/SET/HERE/YOUR/PATH'
SURFACE_PATH_G  = '/SET/HERE/YOUR/PATH'
CHECKPOINT_PATH = '/SET/HERE/YOUR/PATH'
landmarks_g     = []
landmarks_f     = []


def compute_R(lands_g, lands_f):
    # R * X^T = Y
    H = lands_g.transpose(0,1).matmul(lands_f)
    u, e, v = torch.svd(H)
    R = v.matmul(u.transpose(0,1)).detach()

    # check rotation is not a reflection
    if R.det() < 0.0:
        v[:, -1] *= -1
        R = v.matmul(u.transpose(0,1)).detach()
    return R


def main() -> None:

    torch.set_grad_enabled(False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    meta  = SurfaceMapModel()
    net_f = InterMapModel()
    net_q = InterMapModel()

    data_g    = torch.load(SURFACE_PATH_G)
    source    = data_g['grid'].to(device).float()
    faces     = data_g['faces'].long()
    weights_g = data_g['weights']

    data_f    = torch.load(SURFACE_PATH_F)
    weights_f = data_f['weights']
    source_f  = data_f['grid'].to(device).float()

    data_q    = torch.load(SURFACE_PATH_Q)
    weights_q = data_q['weights']
    source_q  = data_q['grid'].to(device).float()

    net_f.load_state_dict(torch.load(CHECKPOINT_PATH_F))
    net_f = net_f.to(device)
    net_q.load_state_dict(torch.load(CHECKPOINT_PATH_Q))
    net_q = net_q.to(device)

    R_f = compute_R(source[landmarks_g], source_f[landmarks_f])
    R_q = compute_R(source[landmarks_g], source_q[landmarks_q])

    for k in weights_g.keys():
        weights_g[k] = weights_g[k].to(device).detach()
        weights_f[k] = weights_f[k].to(device).detach()
        weights_q[k] = weights_q[k].to(device).detach()

    # generate mesh at GT vertices
    G = meta(source, weights_g)
    mapped_fg = net_f(source.matmul(R_f.t()))
    mapped_qg = net_q(source.matmul(R_q.t()))
    F = meta(mapped_fg, weights_f)
    Q = meta(mapped_qg, weights_q)


    show_mesh('G_small.ply', source, G, faces)
    show_mesh('F_small.ply', source, F, faces)
    show_mesh('Q_small.ply', source, Q, faces)


    # generate mesh at sample vertices
    source = data_g['visual_grid'].to(device).float()
    faces  = data_g['visual_faces'].long()

    G = meta(source, weights_g)
    mapped_fg = net_f(source.matmul(R_f.t()))
    mapped_qg = net_q(source.matmul(R_q.t()))
    F = meta(mapped_fg, weights_f)
    Q = meta(mapped_qg, weights_q)

    show_mesh('G_big.ply', source, G, faces)
    show_mesh('F_big.ply', source, F, faces)
    show_mesh('Q_big.ply', source, Q, faces)



if __name__ == '__main__':
    main()
