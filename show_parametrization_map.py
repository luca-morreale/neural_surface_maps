
import torch

from models import SurfaceMapModel
from models import InterMapModel

from utils import show_mesh
from utils import show_mesh_2D


SURFACE_PATH    = '/SET/HERE/YOUR/PATH'
CHECKPOINT_PATH = '/SET/HERE/YOUR/PATH'


def main() -> None:

    torch.set_grad_enabled(False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    meta = SurfaceMapModel()
    net  = InterMapModel()

    data    = torch.load(SURFACE_PATH)
    source  = data['grid'].to(device).float()
    faces   = data['faces'].long()
    weights = data['weights']

    net.load_state_dict(torch.load(CHECKPOINT_PATH))
    net = net.to(device)

    for k in weights.keys():
        weights[k] = weights[k].to(device).detach()

    # generate mesh at GT vertices
    surface = meta(source, weights)
    param   = net(source)


    show_mesh_2D('param.png', param, faces)
    show_mesh('param_small.ply', param, surface, faces)


    # generate mesh at sample vertices
    source = data['visual_grid'].to(device).float()
    faces  = data['visual_faces'].long()

    surface = meta(source, weights)
    param   = net(source)

    show_mesh_2D('param_big.png', param, faces)
    show_mesh('neural_surface_big.ply', param, surface, faces)



if __name__ == '__main__':
    main()
