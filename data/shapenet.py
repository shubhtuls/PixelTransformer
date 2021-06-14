from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path as osp
import h5py
import json
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from . import base_3d


class ShapenetSdfDataset(Dataset):
    def __init__(self, data_dir, category='chair', split='train'):
        self._data_dir = data_dir
        with open(osp.join(data_dir, 'info.json')) as f:
            self._info = json.load(f)
        if category == 'all':
            categories = self._info['all_cats']
        else:
            categories = [category]

        self._model_list = []
        self._synset_list = []
        for c in categories:
            synset = self._info['cats'][c]
            with open(osp.join(data_dir, 'filelists', '{}_{}.lst'.format(synset, split))) as f:
                model_list_s = [l.rstrip('\n') for l in f.readlines()]
                self._model_list += model_list_s
                self._synset_list += [synset]*len(model_list_s)

        np.random.default_rng(seed=0).shuffle(self._model_list)
        np.random.default_rng(seed=0).shuffle(self._synset_list)

    def __len__(self):
        return len(self._model_list)

    def __getitem__(self, index):
        model_id = self._model_list[index]
        synset = self._synset_list[index]
        sdf_h5_file = osp.join(self._data_dir, 'SDF_v1', synset, model_id, 'ori_sample_grid.h5')
        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1,65,65,65)
        return sdf


class ShapenetDataset(base_3d.SDFDataset):
    def __init__(self, data_dir, train=True, category='chair', **kwargs):
        super().__init__(**kwargs)

        split = 'train' if train else 'test'
        self.dset = ShapenetSdfDataset(data_dir, category=category, split=split)