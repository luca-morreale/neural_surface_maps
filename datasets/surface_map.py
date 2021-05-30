
import torch
from .mixin import DatasetMixin

class SurfaceMapDataset(DatasetMixin):

    def __init__(self, config):

        self.sample_path = config.sample_path
        self.num_points  = config.num_points # num points for each iteration
        self.num_epochs  = config.num_epochs # num epochs
        self.sample = self.read_torch_sample(self.sample_path)

        pool_size = self.sample['samples_2d'].size(0)
        self.num_blocks = max(int(pool_size / self.num_points), 1)
        self.blocks = self.split_to_blocks(pool_size, self.num_blocks)


    def __len__(self):
        return self.num_epochs


    def __getitem__(self, index):

        points     = self.sample['points'].float()
        grid       = self.sample['grid'].float()
        samples_2d = self.sample['samples_2d'].float()  # random on grid
        samples_3d = self.sample['samples_3d'].float()  # random on surface
        normals    = self.sample['grid_normals'].float()

        # extract a chunk containing random samples
        idx     = self.blocks[index % self.num_blocks]
        grid    = torch.cat([grid, samples_2d[idx]], dim=0)
        points  = torch.cat([points, samples_3d[idx]], dim=0)
        normals = torch.cat([normals, self.sample['normals'][idx]], dim=0).float()

        data_dict = {
                    'source'       : grid,
                    'gt'           : points,
                    'normals'      : normals,
                    }

        return data_dict
