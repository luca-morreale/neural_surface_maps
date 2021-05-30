
import torch

from models import SurfaceMapModel
from utils import show_mesh


CHECKPOINT_PATH = '/SET/HERE/YOUR/PATH/TO/PTH'


def main() -> None:

    torch.set_grad_enabled(False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = SurfaceMapModel()

    data    = torch.load(CHECKPOINT_PATH)
    source  = data['grid'].to(device).float()
    gt      = data['points'].to(device).float()
    faces   = data['faces'].long()
    weights = data['weights']

    for k in weights.keys():
        weights[k] = weights[k].to(device).detach()

    # generate mesh at GT vertices
    out     = net(source, weights)
    pp_loss = (out - gt).pow(2).sum(-1)

    show_mesh('neural_surface_small.ply', source, out, faces, pp_loss)


    # generate mesh at sample vertices
    source = data['visual_grid'].to(device).float()
    faces  = data['visual_faces'].long()

    out = net(source, weights)

    show_mesh('neural_surface_big.ply', source, out, faces)



if __name__ == '__main__':
    main()
