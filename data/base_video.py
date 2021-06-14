from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import marching_cubes as mcubes
import imageio
import torch
import torchvision
import pdb

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from . import base_img


def sample_video(video, frames, pixels):
    '''
    Args:
        video: nY X nX X H X W X nC
        cams: ... X 1, integers
        pixels: ... X 2, integers
    Returns:
        vals: ... X nC
    '''
    nF, H, W, nC = video.shape

    # following convention in 2D case, we'll assume x is before y in coords
    pixel_x, pixel_y = torch.split(pixels, 1, dim=-1)
    index = frames*(H*W) + pixel_y*(W) + pixel_x
    index = index.long()
    
    vals = []
    for c in range(nC):
        vals.append(torch.take(video[...,c], index))
    return torch.cat(vals, dim=-1)



# -------------- Dataset ------------- #
# ------------------------------------ #
class VideoDataset(Dataset):
    def __init__(self, num_samples=10, num_queries=10, unif_query_sampling=False, n_val_frames=1):
        ## child class should instantiate self.dset
        self.num_samples = num_samples
        self.num_queries = num_queries
        self.unif_query_sampling = unif_query_sampling
        self.n_val_frames = n_val_frames


    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        video = self.dset[index]
        return self._video_to_item(video)

    def _video_to_item(self, video):
        nF, H, W, nC = video.shape

        sample_pixels_y = torch.randint(H, (self.num_samples, 1))
        sample_pixels_x = torch.randint(W, (self.num_samples, 1))

        sample_frames = torch.randint(nF, (self.num_samples, 1))
        sample_pixels = torch.cat((sample_pixels_x, sample_pixels_y), dim=-1)

        if self.unif_query_sampling:
            nq = int(math.sqrt(self.num_queries//self.n_val_frames))
            query_pixels = base_img.coord_grid(nq, nq).view(-1,2)
            query_pixels = base_img.gridify_coords(query_pixels, H, W, normalized=False)
            query_pixels = query_pixels.repeat(self.n_val_frames,1)

            query_frames = base_img.subsample_grid_inds(self.n_val_frames, nF)
            query_frames = query_frames.unsqueeze(1).repeat(1,nq*nq).view(-1,1)
        else:
            query_pixels_y = torch.randint(H, (self.num_queries, 1))
            query_pixels_x = torch.randint(W, (self.num_queries, 1))

            query_frames = torch.randint(nF, (self.num_queries, 1))
            query_pixels = torch.cat((query_pixels_x, query_pixels_y), dim=-1)

        sample_values = sample_video(video, sample_frames, sample_pixels)
        query_values = sample_video(video, query_frames, query_pixels)

        sample_frames = (-1 + 1./nF) + sample_frames*2.0/nF
        query_frames = (-1 + 1./nF) + query_frames*2.0/nF

        sample_pixels = base_img.normalize_coords(sample_pixels, H, W)
        query_pixels = base_img.normalize_coords(query_pixels, H, W)

        sample_positions = torch.cat((sample_frames, sample_pixels), dim=-1)
        query_positions = torch.cat((query_frames, query_pixels), dim=-1)

        return {
            'sample_positions': sample_positions,
            'sample_values': sample_values,
            'query_positions': query_positions,
            'query_values': query_values
        }
