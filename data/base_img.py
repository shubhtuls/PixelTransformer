from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import numpy as np
from PIL import Image


def visualize(positions, val_gt, val_pred=None, sample_positions=None, sample_val=None):
    '''
    Args:
        positions: nS X B X 2 torch tensors
        val_{}: nS X B X nc values
    Returns:
        image array
    '''
    ns = val_gt.shape[0]
    bs = val_gt.shape[1]
    nC = val_gt.shape[2]
    H = int(math.sqrt(ns))
    W = ns // H
    val_gt = torch.clamp(val_gt, 0, 1)
    val_gt = val_gt.view(H,W,bs,nC).permute(2,0,1,3).contiguous()
    val_gt = val_gt.view(-1,W,nC).detach().cpu().numpy()
    vis_img = val_gt

    if val_pred is not None:
        val_pred = torch.clamp(val_pred, 0, 1)
        val_pred = val_pred.view(H,W,bs,nC).permute(2,0,1,3).contiguous()
        val_pred = val_pred.view(-1,W,nC).detach().cpu().numpy()
        vis_img = np.concatenate((val_gt, val_pred), axis=1)

    return vis_img
    

def coord_grid(H, W):
    delta_y = 1/H
    delta_x = 1/W
    xs = torch.linspace(-1 + delta_x, 1 - delta_x, W)
    ys = torch.linspace(-1 + delta_y, 1 - delta_y, H)
    Ys, Xs = torch.meshgrid([ys, xs])

    grid = torch.stack([Xs, Ys], dim=-1)
    return grid


def pixel_grid(H, W):
    '''
    At pixel (i,j), values are (j,i) i.e. grid[y,x] = (x,y)
    '''
    xs = torch.linspace(0,W-1,W)
    ys = torch.linspace(0,H-1,H)
    Ys, Xs = torch.meshgrid([ys, xs])
    grid = torch.stack([Xs, Ys], dim=-1)
    return grid


def normalize_coords(coords, H, W):
    xs, ys = torch.split(coords, 1, dim=-1)
    delta_y = 1./H
    delta_x = 1./W
    xs = (-1 + delta_x) + xs*2.0/W
    ys = (-1 + delta_y) + ys*2.0/H
    
    return torch.cat([xs,ys], dim=-1)


def subsample_grid_inds(ns, D):
    '''
    unfiformly sample ns integers from 0,1,..,D-1
    '''
    delta_ns = 1/ns; delta = 1/D;
    inds = torch.linspace(-1 + delta_ns, 1 - delta_ns, ns)
    inds = torch.round((inds+1-delta)*D/2)
    return inds

    
def gridify_coords(coords, H=128, W=128, normalized=True):
    '''
    Args:
        coords: ... X 2
        H,W: [-1,1] is discretized into H,W grid
    Returns
        coords: ... X 2, discretized coords
    '''
    xs, ys = torch.split(coords, 1, dim=-1)
    delta_y = 1./H
    delta_x = 1./W

    # x = (-1 + dx) + X*2*(1-dx)/(W-1); x=coord, X=pixel
    Xs = (xs + 1 - delta_x)*(W-1)/(2-2*delta_x)
    Ys = (ys + 1 - delta_y)*(H-1)/(2-2*delta_y)

    if normalized:
        xs = (-1 + delta_x) + torch.round(Xs)*2*(1-delta_x)/(W-1)
        ys = (-1 + delta_y) + torch.round(Ys)*2*(1-delta_y)/(H-1)
    else:
        xs = torch.round(Xs)
        ys = torch.round(Ys)

    return torch.cat([xs,ys], dim=-1)
    
    

# -------------- Dataset ------------- #
# ------------------------------------ #
class FolderDataset(Dataset):
    def __init__(self, imgs_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.transform = transform
        self._imgs_list = []
        for root, _, fnames in sorted(os.walk(imgs_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                self._imgs_list.append(path)

    def __len__(self):
        return len(self._imgs_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.imgs_dir, self._imgs_list[index])
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


# -------------- Dataset ------------- #
# ------------------------------------ #
class ImgDataset(Dataset):
    def __init__(self, num_samples=10, num_queries=10, unif_query_sampling=False, grid_sampling=False, imsize=128, out_label=False):
        ## child class should instantiate self.dset
        self.num_samples = num_samples
        self.num_queries = num_queries
        self.unif_query_sampling = unif_query_sampling
        self.grid_sampling = grid_sampling
        self.out_label = out_label
        self.imtransform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((imsize,imsize)), transforms.ToTensor()])

    def _sample_img(self, img, positions):
        '''
        Sample given image at specified positions
        Args:
            img: nC X H X W
            positions: S X 2 samples
        Returns:
            values: S X nC
        '''
        nC = img.shape[0]
        img = img[None,:,:,:]
        positions = positions.view(1,1,-1,2)
        values = torch.nn.functional.grid_sample(img, positions, align_corners=False)
        values = values.view(nC,-1).permute(1,0).contiguous()
        return values
    
    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        if self.out_label:
            img, label = self.dset[index]
        else:
            img = self.dset[index]

        sample_positions = (torch.rand(self.num_samples, 2)*2 - 1)
        if self.grid_sampling:
            sample_positions = gridify_coords(sample_positions)
        sample_values = self._sample_img(img, sample_positions)

        if self.unif_query_sampling:
            nq = int(math.sqrt(self.num_queries))
            query_positions = coord_grid(nq, nq).view(-1,2)
        else:
            query_positions = (torch.rand(self.num_queries, 2)*2 - 1)

        if self.grid_sampling:
            query_positions = gridify_coords(query_positions)
        query_values = self._sample_img(img, query_positions)

        elem = {
            'img': self.imtransform(img),
            'sample_positions': sample_positions,
            'sample_values': sample_values,
            'query_positions': query_positions,
            'query_values': query_values
        }
        if self.out_label:
            elem['label'] = label

        return elem
