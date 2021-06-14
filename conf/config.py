from dataclasses import dataclass, field
from typing import Any, List
from omegaconf import MISSING, OmegaConf

import hydra
import os.path as osp
import pdb
from hydra.core.config_store import ConfigStore

from . import data as data_config

cs = ConfigStore.instance()
data_config.add_confs(cs)

defaults = [
    {"data": "cifar"},
    {"eval": "default"},
    {"eval.checkpoint": "default"},
    {"logging": "tensorboard"},
    {"model": "default"},
    {"model.transformer": "default"},
    {"model.posnenc": "fourier"},
    {"optim": "default"},
    {"optim.pretrain_checkpoint": "default"},
    {"resources": "default"},
]


@dataclass
class CheckpointConfig:
    name: Any = None
    version: int = 0
    epoch: Any = 'last'


def extract_ckpt_path(cfg):
    path = osp.join(cfg.name, 'version_{}'.format(cfg.version))
    if cfg.epoch == 'last':
        checkpoint_path = osp.join(path, 'checkpoints', 'last.ckpt')
    else:
        checkpoint_path = osp.join(path, 'checkpoints', 'epoch={}.ckpt'.format(cfg.epoch))
    return checkpoint_path


@dataclass
class EvalConfig:
    checkpoint: CheckpointConfig = MISSING
    eval_dir: str = 'cachedir/eval'
    n_iter: int = 100  # number of instances to evaluate
    ns_min: int = 4  # min number of pixel samples
    ns_max: int = 2048  # max number of samples
    ns_steps: int = 10   # number of total sample sizes including min and max
    n_cond_samples: int = 5  # number of conditional images generated
    tau: float = 0.9  # scaling factor for conditional proabilities before sampling. large tau implies more outliers
    classify: bool = False # set to true for cifar classification accuracy plots


cs.store(group="eval", name="default", node=EvalConfig)
cs.store(group="eval.checkpoint", name="default", node=CheckpointConfig)


@dataclass
class LoggingConfig:
    log_dir: str = 'cachedir/tensorboard_logs'
    name: str = MISSING
cs.store(group="logging", name="tensorboard", node=LoggingConfig)



@dataclass
class TransformerConfig:
    # network hyperparams
    num_decoder_layers: int = 10
    num_encoder_layers: int = 10
    nhead: int = 4
    prenorm: bool = True


@dataclass
class PosnencFourierConfig:
    # hyperparams for encoding of input positions
    mode: str = 'fourier'
    init_factor: int = 2


@dataclass
class PosnencLinearConfig:
    mode: str = 'linear'


@dataclass
class ModelConfig:
    nde: int = 128  # dimension of per-sample latent embedding
    posnenc: Any = MISSING
    transformer: TransformerConfig = MISSING
    # format of output distribution. use 'std' for polynomial experiments, 'disccont' otherwise
    # disccont = dicrete + continuous, and requires a file to specify the precomputed discrete bins 
    out_pde: str = 'disccont'


cs.store(group="model", name="default", node=ModelConfig)
cs.store(group="model.transformer", name="default", node=TransformerConfig)
cs.store(group="model.posnenc", name="fourier", node=PosnencFourierConfig)
cs.store(group="model.posnenc", name="linear", node=PosnencLinearConfig)


@dataclass
class OptimizationConfig:
    val_check_interval: int = 200
    num_val_iter: int = 1
    save_freq: int = 1
    max_epochs: int = 3000
    lr: float = 0.0001
    use_pretrain: bool = False
    pretrain_checkpoint: CheckpointConfig = MISSING
cs.store(group="optim", name="default", node=OptimizationConfig)
cs.store(group="optim.pretrain_checkpoint", name="default", node=CheckpointConfig)


@dataclass
class ResourceConfig:
    gpus: int = 1
    num_workers: int = 0
cs.store(group="resources", name="default", node=ResourceConfig)


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    data: data_config.DataConfig = MISSING
    eval: EvalConfig = MISSING
    logging: LoggingConfig = MISSING
    model: ModelConfig = MISSING
    optim: OptimizationConfig = MISSING
    resources: ResourceConfig = MISSING


cs.store(name="config", node=Config)