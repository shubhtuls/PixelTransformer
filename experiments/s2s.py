"""
Sample usage:
      python -m PixelTransformer.experiments.s2s logging.name=debug data=polynomial data.degree=6 model.out_pde=std
      python -m PixelTransformer.experiments.s2s logging.name=debug data=cats      

# python ~/scripts/cluster_run.py --setup=/private/home/shubhtuls/scripts/init_s2s_cluster.sh --partition='dev' --cmd='' --max_mem
"""

import os
import os.path as osp
_curr_path = osp.dirname(osp.abspath(__file__))
_base_path = osp.join(_curr_path, '..')

from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
import pdb
import numpy as np

import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler

from ..conf import config
from ..nnutils import model as model_util
from ..nnutils import subsampling as subsampling_util
from ..nnutils import nn_model
from ..data import dataset_manager
from ..nnutils import distributions as distribution_util


class S2SModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        out_pde_fn, ndv_out = distribution_util.pde_router(cfg.model.out_pde, cfg.data.ndv, cfg.data.cluster_centres)

        self.model = model_util.S2SNet(
            cfg.data.ndp, cfg.data.ndv, cfg.model.nde,
            cfg.model.transformer, cfg.model.posnenc,
            ndv_out=ndv_out, out_pde_fn=out_pde_fn,
        )
        
        self.ns_sampler = subsampling_util.subsampling_manager(cfg.data.ss_mode, cfg.data.max_subsampling)
        self.nn_model = nn_model.NNNet()

    def train_dataloader(self):
        cfg = self.cfg
        dset = dataset_manager.train_dataset(cfg.data)
        dl = torch.utils.data.DataLoader(dset, batch_size=cfg.data.bs_train, num_workers=cfg.resources.num_workers)    
        return dl

    def val_dataloader(self):
        cfg = self.cfg
        dset, self.pred_vis_fn = dataset_manager.val_dataset(cfg.data)
        dset = dataset_manager.DatasetPermutationWrapper(dset)
        dl = torch.utils.data.DataLoader(dset, batch_size=cfg.data.bs_val, num_workers=cfg.resources.num_workers)
        return dl

    def forward(self, sample_val, sample_pos, query_pos):
        return self.model(sample_val, sample_pos, query_pos)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr)

    def extract_batch_input(self, batch, sample_variable=False):
        sample_pos = batch['sample_positions'].permute(1,0,2).contiguous()
        sample_val = batch['sample_values'].permute(1,0,2).contiguous()

        if self.cfg.data.max_subsampling > 1.0:
            if sample_variable:
                sample_frac = self.ns_sampler.sample()
            else:
                sample_frac = self.ns_sampler.median()
            ns = int(np.ceil(sample_pos.shape[0]*sample_frac))
            sample_pos = sample_pos[0:ns]
            sample_val = sample_val[0:ns]

        query_pos = batch['query_positions'].permute(1,0,2).contiguous()
        query_val = batch['query_values'].permute(1,0,2).contiguous()
        return sample_pos, sample_val, query_pos, query_val

    def validation_step(self, batch, batch_idx):
        sample_pos, sample_val, query_pos, query_val = self.extract_batch_input(batch)
        with torch.no_grad():
            pred = self(sample_val, sample_pos, query_pos)
            loss = pred.nll(query_val).mean()
        vis_img = self.pred_vis_fn(
            query_pos, query_val, val_pred=pred.mean(), sample_positions=sample_pos, sample_val=sample_val)
        self.logger.experiment.add_image('results', vis_img.transpose((2,0,1)))
        if self.cfg.model.out_pde == 'discrete':
            vis_img = self.pred_vis_fn(query_pos, pred.discretize(query_val))
            self.logger.experiment.add_image('gt_discretized', vis_img.transpose((2,0,1)))

        pred_nn = self.nn_model(sample_val, sample_pos, query_pos)
        vis_img_nn = self.pred_vis_fn(
            query_pos, query_val, val_pred=pred_nn, sample_positions=sample_pos, sample_val=sample_val)
        self.logger.experiment.add_image('results_nn', vis_img_nn.transpose((2,0,1)))

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs}

    def training_step(self, batch, batch_idx):
        sample_pos, sample_val, query_pos, query_val = self.extract_batch_input(batch, sample_variable=True)
        pred = self(sample_val, sample_pos, query_pos)
        loss = pred.nll(query_val).mean()

        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}


@hydra.main(config_name="config")
def main(cfg: config.Config) -> None:
    model = S2SModel(cfg=cfg)
    log_dir = osp.join(_base_path, cfg.logging.log_dir, cfg.logging.name)
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(cfg, osp.join(log_dir, 'config.txt'))

    logger = TensorBoardLogger(osp.join(_base_path, cfg.logging.log_dir), name=cfg.logging.name)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_val_epochs=cfg.optim.save_freq,
        save_last=True
    )

    if cfg.optim.use_pretrain:
        checkpoint_cfg = cfg.optim.pretrain_checkpoint
        checkpoint_path = config.extract_ckpt_path(checkpoint_cfg)
        checkpoint = osp.join(_base_path, cfg.logging.log_dir, checkpoint_path)
    else:
        checkpoint = None

    trainer = Trainer(
        logger=logger,
        gpus=cfg.resources.gpus,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.num_val_iter,
        callbacks = [checkpoint_callback],
        resume_from_checkpoint=checkpoint,
        max_epochs=cfg.optim.max_epochs
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()