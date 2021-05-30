
import pickle
import torch

class DatasetMixin:

    def read_pickle_sample(self, path):
        '''
            Load file located at `path`. File assumed to be a pickle binary.
        '''
        return pickle.load(open(path, 'rb'))

    def read_torch_sample(self, path):
        '''
            Load file located at `path`. File assumed to be a torch binary.
        '''
        return torch.load(path, map_location='cpu')


    def read_map_sample(self, path):
        '''
            Read a file at `path` containing a surface map.
            By default weights representing the map are moved to CPU and
            flagged as not trainable. Faces for visualization are also
            converted into long.
        '''
        sample = self.read_torch_sample(path)

        sample['faces'] = sample['faces'].long()
        sample['visual_faces'] = sample['visual_faces'].long()
        sample['C'] = None if 'C' not in sample else sample['C']

        for k in sample['weights'].keys():
            sample['weights'][k].requires_grad = False
            sample['weights'][k] = sample['weights'][k].cpu()

        return sample

    def split_to_blocks(self, size, num_blocks):
        '''
            Split the set of possible points into chuncks.
            Then permutes the indices to have random sampling.
        '''
        idxs = torch.randperm(size)
        block_size = int(float(idxs.size(0)) / float(num_blocks))
        blocks = []
        for i in range(num_blocks):
            blocks.append(idxs[block_size * i : block_size * (i + 1)])

        return blocks

    def compute_lands_rotation(self, lands_g, lands_f):
        '''
            Compute a rotation matrix aligning two set of landmarks.
        '''
        with torch.no_grad(): # not sure if this is necessary
            # R * X^T = Y
            H = lands_g.transpose(0,1).matmul(lands_f)
            u, e, v = torch.svd(H)
            R = v.matmul(u.transpose(0,1)).detach()

            # fix rotation if it is a reflection
            if R.det() < 0.0:
                v[:, -1] *= -1
                R = v.matmul(u.transpose(0,1)).detach()

        return R
