from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset

from . import base_img


class MNISTImageWrapper(Dataset):
    def __init__(self, data_dir, **kwargs):
        self.dset = MNIST(data_dir, **kwargs)
    
    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        img = self.dset[index][0]
        return img


class MNISTDataset(base_img.ImgDataset):
    def __init__(self, data_dir, train=True, **kwargs):
        super().__init__(**kwargs)

        self.dset = MNISTImageWrapper(data_dir, train=train, download=False, transform=transforms.ToTensor())