from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import marching_cubes as mcubes
import imageio
import torch
import torchvision
import os.path as osp
import os
import time
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from . import base_video


class BeachVideosDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self._data_dir = osp.join(data_dir, 'frames-stable-many')
        with open(osp.join(data_dir, 'beach_{}.txt'.format(split))) as f:
            self._video_list = [l.rstrip() for l in f.readlines()]

        np.random.default_rng(seed=0).shuffle(self._video_list)
        self.transform = transforms.ToTensor()
        self.split = split
    
    def __len__(self):
        return len(self._video_list)
    
    def __getitem__(self, index):
        clip_file = osp.join(self._data_dir, self._video_list[index])
        if not osp.exists(clip_file):
            # sometimes does not exist - not sure why!
            index = np.random.randint(len(self._video_list))
            return self.__getitem__(index)

        with open(clip_file, 'rb') as f:
            im = Image.open(f)
            im = im.convert('RGB')
            im = self.transform(im).permute(1,2,0)

        if im.shape[0] == 34*128:            
            im = im.reshape((34,128,128,3))
            return im
        else:
            # occasionally, last clip of video is short
            # in this case return penultimate clip
            return self.__getitem__(index-1)
    

class BeachDataset(base_video.VideoDataset):
    def __init__(self, data_dir, split='train', **kwargs):
        super().__init__(**kwargs)

        self.dset = BeachVideosDataset(data_dir, split=split)

