from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import Dataset

from . import base_img


class CelebAImageWrapper(Dataset):
    def __init__(self, data_dir, **kwargs):
        self.dset = CelebA(data_dir, **kwargs)
    
    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        img = self.dset[index][0]
        return img


class CelebADataset(base_img.ImgDataset):
    def __init__(self, data_dir, split='train', **kwargs):
        super().__init__(**kwargs)

        transform = transforms.Compose(
            [transforms.Resize(128),
             transforms.CenterCrop(128),
             transforms.ToTensor()])
        self.dset = CelebAImageWrapper(data_dir, split=split, download=False, transform=transforms.ToTensor())
        # self.dset = CelebAImageWrapper(data_dir, split=split, download=False, transform=transform)
