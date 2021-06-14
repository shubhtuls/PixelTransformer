from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pdb
from torch.utils.data import Dataset

from . import polynomial as poly_data
from . import base_img as base_img_data
from . import base_3d as base_3d_data
from . import beach as beach_data
from . import cifar as cifar_data
from . import cats as cats_data
from . import celeba as celeba_data
from . import mnist as mnist_data
from . import shapenet as shapenet_data


def train_dataset(data_cfg):
    base_args = {'num_samples': data_cfg.ns, 'num_queries': data_cfg.nq}
    if 'data_dir' in data_cfg:
        base_args['data_dir'] = data_cfg.data_dir
    if 'grid_sampling' in data_cfg:
        base_args['grid_sampling'] = data_cfg.grid_sampling

    if data_cfg.dataset == 'polynomial':
        dset = poly_data.PolynomialDataset(degree=data_cfg.degree, **base_args)
    elif data_cfg.dataset == 'beach':
        dset = beach_data.BeachDataset(split='train', **base_args)
    elif data_cfg.dataset == 'cats':
        dset = cats_data.CatsDataset(split='train', **base_args)
    elif data_cfg.dataset == 'celeba':
        dset = celeba_data.CelebADataset(split='train', **base_args)
    elif data_cfg.dataset == 'cifar100':
        dset = cifar_data.CIFAR100Dataset(train=True, **base_args)
    elif data_cfg.dataset == 'cifar10':
        dset = cifar_data.CIFAR10Dataset(train=True, **base_args)
    elif data_cfg.dataset == 'cifar10label':
        dset = cifar_data.CIFAR10LabelDataset(train=True, **base_args)
    elif data_cfg.dataset == 'shapenet':
        dset = shapenet_data.ShapenetDataset(category=data_cfg.category, train=True, **base_args)
    else:
        raise ValueError('Unsupported dataset!')
    return dset


def val_dataset(data_cfg):
    base_args = {'num_samples': data_cfg.ns, 'num_queries': data_cfg.nq_val, 'unif_query_sampling': True}
    if 'data_dir' in data_cfg:
        base_args['data_dir'] = data_cfg.data_dir
    if 'grid_sampling' in data_cfg:
        base_args['grid_sampling'] = data_cfg.grid_sampling

    if data_cfg.dataset == 'polynomial':
        pred_vis_fn = poly_data.visualize
        dset = poly_data.PolynomialDataset(degree=data_cfg.degree, **base_args)
    elif data_cfg.dataset == 'beach':
        pred_vis_fn = base_img_data.visualize
        dset = beach_data.BeachDataset(split='val', n_val_frames=data_cfg.n_val_frames, **base_args)
    elif data_cfg.dataset == 'cats':
        pred_vis_fn = base_img_data.visualize
        dset = cats_data.CatsDataset(split='val', **base_args)
    elif data_cfg.dataset == 'celeba':
        pred_vis_fn = base_img_data.visualize
        dset = celeba_data.CelebADataset(split='valid', **base_args)
    elif data_cfg.dataset == 'cifar100':
        pred_vis_fn = base_img_data.visualize
        dset = cifar_data.CIFAR100Dataset(train=False, **base_args)
    elif data_cfg.dataset == 'cifar10label':
        pred_vis_fn = base_img_data.visualize
        dset = cifar_data.CIFAR10LabelDataset(train=False, **base_args)
    elif data_cfg.dataset == 'cifar10':
        pred_vis_fn = base_img_data.visualize
        dset = cifar_data.CIFAR10Dataset(train=False, **base_args)
    elif data_cfg.dataset == 'mnist':
        pred_vis_fn = base_img_data.visualize
        dset = mnist_data.MNISTDataset(train=False, **base_args)
    elif data_cfg.dataset == 'shapenet':
        pred_vis_fn = base_3d_data.visualize
        dset = shapenet_data.ShapenetDataset(category=data_cfg.category,
            train=False, nz_slices=data_cfg.nz_slices, **base_args)
    else:
        raise ValueError('Unsupported dataset!')
    return dset, pred_vis_fn


class DatasetPermutationWrapper(Dataset):
    def __init__(self, dset):
        self.dset = dset
        self._len = len(self.dset)
    
    def __len__(self):
        return self._len

    def __getitem__(self, _):
        index = np.random.randint(self._len)
        return self.dset[index]
