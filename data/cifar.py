from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10

from torchvision import transforms
from torch.utils.data import Dataset

from . import base_img


class CIFAR100ImageWrapper(Dataset):
    def __init__(self, data_dir, **kwargs):
        self.dset = CIFAR100(data_dir, **kwargs)
    
    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        img = self.dset[index][0]
        return img


class CIFAR10ImageWrapper(Dataset):
    def __init__(self, data_dir, **kwargs):
        self.dset = CIFAR10(data_dir, **kwargs)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        img = self.dset[index][0]
        return img


class CIFAR100Dataset(base_img.ImgDataset):
    def __init__(self, data_dir, train=True, **kwargs):
        super().__init__(**kwargs)

        self.dset = CIFAR100ImageWrapper(data_dir, train=train, download=False, transform=transforms.ToTensor())


class CIFAR10Dataset(base_img.ImgDataset):
    def __init__(self, data_dir, train=True, **kwargs):
        super().__init__(**kwargs)

        self.dset = CIFAR10ImageWrapper(data_dir, train=train, download=False, transform=transforms.ToTensor())

        
class CIFAR10LabelDataset(base_img.ImgDataset):
    def __init__(self, data_dir, train=True, **kwargs):
        super().__init__(out_label=True, imsize=32, **kwargs)

        self.dset = CIFAR10(data_dir, train=train, download=False, transform=transforms.ToTensor())

