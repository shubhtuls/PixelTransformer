from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    bs_train: int = 6
    bs_val: int = 4
    # 'log' - > subsampling will be linear in log space, 'linear' -> 
    ss_mode: str = "log"
    cluster_centres: str = ''
    dataset: str = MISSING


@dataclass
class ImgDataConfig(DataConfig):
    ndv: int = 3
    ndp: int = 2
    ns: int = 2048
    nq: int = 2048
    nq_val: int = 4096
    max_subsampling: int = 128
    cluster_centres:  str = '/private/home/shubhtuls/code/s2s/cachedir/bins/img.npy'
    grid_sampling:  bool = False


@dataclass
class CatsDataConfig(ImgDataConfig):
    dataset: str = 'cats'
    data_dir: str = '/private/home/shubhtuls/code/s2s/cachedir/cats/cat_combined'


@dataclass
class CelebaDataConfig(ImgDataConfig):
    dataset: str = 'celeba'
    data_dir: str = '/private/home/shubhtuls/code/s2s/cachedir/celebA'


@dataclass
class Cifar100DataConfig(ImgDataConfig):
    dataset: str = 'cifar100'
    data_dir: str = '/private/home/shubhtuls/code/s2s/cachedir/cifar100'


@dataclass
class Cifar10DataConfig(ImgDataConfig):
    dataset: str = 'cifar10'
    data_dir: str = '/private/home/shubhtuls/code/s2s/cachedir/cifar10'


@dataclass
class Cifar10LabelDataConfig(Cifar10DataConfig):
    dataset: str = 'cifar10label'


@dataclass
class BeachDataConfig(DataConfig):
    ndv: int = 3
    ndp: int = 3
    ns: int = 2048
    nq: int = 2048
    nq_val: int = 4096
    max_subsampling: int = 4
    cluster_centres:  str = '/private/home/shubhtuls/code/s2s/cachedir/bins/img.npy'
    dataset: str = 'beach'
    data_dir: str =  '/private/home/shubhtuls/code/s2s/cachedir/videos/'
    n_val_frames: int = 1


@dataclass
class MnistDataConfig(ImgDataConfig):
    dataset: str = 'mnist'
    data_dir: str = '/private/home/shubhtuls/code/s2s/cachedir/mnist'
    ndv: int = 1
    ns: int = 512
    nq: int = 512
    nq_val: int = 784
    max_subsampling: int = 64
    cluster_centres:  str = '/private/home/shubhtuls/code/s2s/cachedir/bins/mnist.npy'


@dataclass
class PolynomialDataConfig(DataConfig):
    dataset: str = 'polynomial'
    degree: int = 6
    ndv: int = 1
    ndp: int = 1
    ns: int = 20
    nq: int = 20
    nq_val: int = 100
    max_subsampling: int = 4


@dataclass
class ShapenetDataConfig(DataConfig):
    dataset: str = 'shapenet'
    category: str = 'chair'
    data_dir: str = '/private/home/shubhtuls/code/s2s/cachedir/shapenet'
    cluster_centres:  str = '/private/home/shubhtuls/code/s2s/cachedir/bins/shapenet.npy'
    ndv: int = 1
    ndp: int = 3
    ns: int = 2048
    nq: int = 2048
    nq_val: int = 4096
    max_subsampling: int = 128
    nz_slices: int = 4


def add_confs(cs):
    cs.store(group="data", name="beach", node=BeachDataConfig)
    cs.store(group="data", name="cats", node=CatsDataConfig)
    cs.store(group="data", name="cifar100", node=Cifar100DataConfig)
    cs.store(group="data", name="cifar10", node=Cifar10DataConfig)
    cs.store(group="data", name="cifar10label", node=Cifar10LabelDataConfig)
    cs.store(group="data", name="celeba", node=CelebaDataConfig)
    cs.store(group="data", name="mnist", node=MnistDataConfig)
    cs.store(group="data", name="polynomial", node=PolynomialDataConfig)
    cs.store(group="data", name="shapenet", node=ShapenetDataConfig)