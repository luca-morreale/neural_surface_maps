
import trimesh

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

mpl.rcParams['agg.path.chunksize'] = 10000


def show_mesh(filename, source, out, faces, pp_loss=None):
    np_uvs   = source.cpu().numpy()
    np_verts = out.cpu().numpy()
    np_faces = faces.cpu().numpy()

    mesh = trimesh.Trimesh(
                    vertices=np_verts,
                    faces=np_faces,
                    vertex_attributes={
                        'texture_u': np_uvs[:,0], # for meshlab
                        'texture_v': np_uvs[:,1], # for meshlab
                        's': np_uvs[:,0], # for blender
                        't': np_uvs[:,1], # for blender
                        },
                    process=False) # no data reordering

    if pp_loss is not None:
        mesh.vertex_attributes['error'] = pp_loss.cpu().numpy()

    mesh.export(filename)


def show_mesh_2D(filename, uv_points, triangles, landmarks=None):
    uv_points = uv_points.detach().cpu().numpy()
    triangles = triangles.detach().cpu().numpy()

    # draw image of conformal points given faces
    plt.figure(figsize=(10, 10), dpi=90)
    plt.title('Mesh layout')

    plt.triplot(uv_points[:,0], uv_points[:,1], triangles, linewidth=0.5, c='k')

    plt.axis('equal')
    plt.savefig(filename)
    plt.close()

