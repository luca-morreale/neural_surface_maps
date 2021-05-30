
import torch
from .mixin import DatasetMixin

class ParametrizationDataset(DatasetMixin):

    def __init__(self, config):
        super().__init__()

        self.sample_path = config.sample_path
        self.num_points  = config.num_points
        self.num_epochs  = config.num_epochs

        sample = self.read_map_sample(self.sample_path)

        self.source     = sample['grid'].float()
        self.faces      = sample['faces'].long()
        self.weights    = sample['weights']
        self.pool       = sample['visual_grid'].float()

        # split data into chuncks
        self.num_blocks  = int(self.pool.size(0) / self.num_points)
        self.num_blocks  = self.num_blocks if self.num_blocks > 1 else 1
        self.blocks_idxs = self.split_to_blocks(self.pool.size(0), self.num_blocks)


    def __len__(self):
        return self.num_epochs


    def __getitem__(self, index):

        # extract a chunk containing random samples
        idxs = self.blocks_idxs[index % self.num_blocks]
        source = self.pool[idxs]

        data_dict = {'source'        : source,
                    'weights'        : self.weights,
                    }

        return data_dict
