
import torch
from .inter_map import MapDataset

class CollectionDataset(MapDataset):

    def __init__(self, config):
        super().__init__(config)
        self.path_q     = config.sample_path_q

        sample_q = self.read_map_sample(self.path_q)

        self.source_q  = sample_q['grid'].float()
        self.weights_q = sample_q['weights']
        self.C_q       = sample_q['C']

        self.lands_q = self.source_q[config.landmarks_q]

        self.R_q = self.compute_lands_rotation(self.lands_g, self.lands_q)

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, index):

        # get a chuck with random points
        idx = index % self.num_blocks
        mesh_source_g = self.pool_g[self.blocks_g[idx]]
        # blocks_g has been set in MapDataset


        data_dict = {
                    'source_g'        : mesh_source_g,
                    'weights_g'       : self.weights_g,
                    'weights_f'       : self.weights_f,
                    'weights_q'       : self.weights_q,
                    'C_g'             : self.C_g,
                    'C_f'             : self.C_f,
                    'C_q'             : self.C_q,
                    'lands_g'         : self.lands_g,
                    'lands_f'         : self.lands_f,
                    'lands_q'         : self.lands_q,
                    'boundary_g'      : self.boundary,
                    'R_f'             : self.R,
                    'R_q'             : self.R_q,
                    }

        return data_dict
