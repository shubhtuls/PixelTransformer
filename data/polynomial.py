from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def visualize(positions, val_gt, val_pred=None, sample_positions=None, sample_val=None):
    '''
    Args:
        positions, val_{}: nS X B X 1 torch tensors
    Returns:
        image array
    '''
    positions = positions.detach().cpu().numpy()
    val_gt = val_gt.detach().cpu().numpy()
    if val_pred is not None:
        val_pred = val_pred.detach().cpu().numpy()
    if sample_positions is not None:
        sample_positions = sample_positions.detach().cpu().numpy()
        sample_val = sample_val.detach().cpu().numpy()
    bs = positions.shape[1]


    fig = plt.figure()
    for b in range(bs):
        fig.add_subplot(2, (bs+1)//2, b+1)
        plt.plot(positions[:,b,0], val_gt[:,b,0], 'g')
        if val_pred is not None:
            plt.plot(positions[:,b,0], val_pred[:,b,0], 'b')
        if sample_positions is not None:
            plt.scatter(sample_positions[:,b,0], sample_val[:,b,0], c='r')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

# -------------- Dataset ------------- #
# ------------------------------------ #
class PolynomialDataset(Dataset):
    def __init__(self, degree=2, num_samples=10, num_queries=10, unif_query_sampling=False):
        self.degree = degree
        self.num_samples = num_samples
        self.num_queries = num_queries
        self.unif_query_sampling = unif_query_sampling
    
    def _sample_coeffs(self):
        return torch.randn(self.degree+1)
    
    def _evaluate(self, coefficients, positions):
        '''
        Evaluate given polynomial at specified positions
        Args:
            coefficients: D dim
            positions: S samples
        Returns:
            values: S
        '''
        x_powers = []
        x_pow_d = torch.ones_like(positions)
        for d in range(self.degree+1):
            x_powers.append(x_pow_d)
            x_pow_d = x_pow_d*positions

        # S X D
        x_powers = torch.stack(x_powers, dim=1)
        values = torch.sum(x_powers*coefficients, dim=1)
        return values
    
    def __len__(self):
        return int(1e6)

    def __getitem__(self, index):
        coefficients = self._sample_coeffs()

        sample_positions = (torch.rand(self.num_samples)*2 - 1)
        sample_values = self._evaluate(coefficients, sample_positions)

        if self.unif_query_sampling:
            query_positions = torch.linspace(-1, 1, self.num_queries)
        else:
            query_positions = (torch.rand(self.num_queries)*2 - 1)
        query_values = self._evaluate(coefficients, query_positions)

        return {
            'coefficients': coefficients,
            'sample_positions': sample_positions.view(-1,1),
            'sample_values': sample_values.view(-1,1),
            'query_positions': query_positions.view(-1,1),
            'query_values': query_values.view(-1,1)
        }
