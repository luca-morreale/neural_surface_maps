
import torch
from .mixin import DatasetMixin

class MapDataset(DatasetMixin):

    def __init__(self, config):
        self.config     = config
        self.path_g     = config.sample_path_g
        self.path_f     = config.sample_path_f
        self.num_points = config.num_points
        self.num_epochs = config.num_epochs

        sample_g = self.read_map_sample(config.sample_path_g)
        sample_f = self.read_map_sample(config.sample_path_f)

        # save data from sample_g
        self.source_g     = sample_g['grid'].float()
        self.faces_g      = sample_g['faces']
        self.weights_g    = sample_g['weights']
        self.C_g          = sample_g['C']
        self.pool_g       = torch.cat([sample_g['grid'], sample_g['samples_2d']], dim=0).float()
        self.visual_grid  = sample_g['visual_grid'].float()
        self.visual_faces = sample_g['visual_faces']
        self.boundary     = sample_g['boundary'].float()

        # save data from sample_f
        self.source_f     = sample_f['grid']
        self.weights_f    = sample_f['weights']
        self.C_f          = sample_f['C']

        # split data into chuncks
        self.num_blocks = max(int(self.pool_g.size(0) / self.num_points), 0)
        self.blocks_g   = self.split_to_blocks(self.pool_g.size(0), self.num_blocks)

        # save landmarks locations
        self.lands_g = self.source_g[config.landmarks_g].float()
        self.lands_f = self.source_f[config.landmarks_f].float()

        self.R = self.compute_lands_rotation(self.lands_g, self.lands_f)

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, index):

        # extract a chunk containing random samples
        idx = index % self.num_blocks
        mesh_source_g = self.pool_g[self.blocks_g[idx]]


        data_dict = {
                    'source_g'        : mesh_source_g,
                    'weights_g'       : self.weights_g,
                    'weights_f'       : self.weights_f,
                    'C_g'             : self.C_g,
                    'C_f'             : self.C_f,
                    'lands_g'         : self.lands_g,
                    'lands_f'         : self.lands_f,
                    'boundary_g'      : self.boundary,
                    'R'               : self.R,
                    }

        return data_dict
