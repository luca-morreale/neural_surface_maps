
import torch

from models import SurfaceMapModel
from models import InterMapModel

from utils import show_mesh
from utils import show_mesh_2D


SURFACE_PATH_F  = '/SET/HERE/YOUR/PATH'
SURFACE_PATH_G  = '/SET/HERE/YOUR/PATH'
CHECKPOINT_PATH = '/SET/HERE/YOUR/PATH'
landmarks_g     = []
landmarks_f     = []


def compute_R(lands_g, lands_f):
    centered_g = lands_g - lands_g.mean(dim=0)
    centered_f = lands_f - lands_f.mean(dim=0)
    # R * X^T = Y
    H = centered_g.transpose(0,1).matmul(centered_f)
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

    meta = SurfaceMapModel()
    net  = InterMapModel()

    data_g    = torch.load(SURFACE_PATH_G)
    source    = data_g['grid'].to(device).float()
    faces     = data_g['faces'].long()
    weights_g = data_g['weights']

    data_f    = torch.load(SURFACE_PATH_F)
    weights_f = data_f['weights']
    source_f  = data_f['grid'].to(device).float()

    net.load_state_dict(torch.load(CHECKPOINT_PATH))
    net = net.to(device)

    R = compute_R(source[landmarks_g], source_f[landmarks_f])

    for k in weights_g.keys():
        weights_g[k] = weights_g[k].to(device).detach()
        weights_f[k] = weights_f[k].to(device).detach()

    # generate mesh at GT vertices
    G = meta(source, weights_g)
    mapped_g = net(source.matmul(R.t()))
    F = meta(mapped_g, weights_f)


    show_mesh('G_small.ply', source, G, faces)
    show_mesh('F_small.ply', source, F, faces)


    # generate mesh at sample vertices
    source = data_g['visual_grid'].to(device).float()
    faces  = data_g['visual_faces'].long()

    G = meta(source, weights_g)
    mapped_g = net(source.matmul(R.t()))
    F = meta(mapped_g, weights_f)

    show_mesh('G_big.ply', source, G, faces)
    show_mesh('F_big.ply', source, F, faces)



if __name__ == '__main__':
    main()
