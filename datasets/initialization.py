
import torch

class InitalizationDataset:

    def __init__(self, num_points=10000, num_epochs=30000):

        self.num_points  = num_points # num points for each iteration
        self.num_epochs  = num_epochs # num epochs


    def __len__(self):
        return self.num_epochs


    def __getitem__(self, index):

        # randomly sample `num_points` points between -10.0 and 10.0
        samples = torch.FloatTensor(self.num_points, 2).uniform_(-10.0, 10.0)

        return samples
