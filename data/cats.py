from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchvision import transforms
from torch.utils.data import Dataset
import os.path as osp

from . import base_img

class CatsDataset(base_img.ImgDataset):
    def __init__(self, data_dir, split='train', **kwargs):
        super().__init__(**kwargs)
        self.dset = base_img.FolderDataset(osp.join(data_dir, split), transform=transforms.ToTensor())
