'''
Base classes for defining a tester
'''

import os
import os.path as osp
import glob

from omegaconf import DictConfig
import hydra
import pdb
import numpy as np

import torch

from pytorch_lightning.core.lightning import LightningModule

from ..conf import config
from ..nnutils import model as model_util
from ..nnutils import nn_model
from ..data import dataset_manager
from ..nnutils import distributions as distribution_util


class S2SModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        out_pde_fn, ndv_out = distribution_util.pde_router(cfg.model.out_pde, cfg.data.ndv, cfg.data.cluster_centres)
        self.model = self.create_model(cfg, out_pde_fn, ndv_out)
        self.nn_model = nn_model.NNNet()


    def create_model(self, cfg, out_pde_fn, ndv_out):
        return model_util.S2SNet(
            cfg.data.ndp, cfg.data.ndv, cfg.model.nde,
            cfg.model.transformer, cfg.model.posnenc,
            ndv_out=ndv_out, out_pde_fn=out_pde_fn,
        )

    def forward(self, sample_val, sample_pos, query_pos):
        return self.model(sample_val, sample_pos, query_pos).mean()

    def forward_sample(self, sample_val, sample_pos, query_pos):
        tau=self.cfg.eval.tau
        return self.model(sample_val, sample_pos, query_pos).sample(tau=tau)

    def forward_conditional(self, sample_val, sample_pos, query_pos, permute_order=True, return_perm=False, perm_generator=None):
        max_cond_samples = self.cfg.data.ns//2
        tau=self.cfg.eval.tau

        if permute_order:
            nq = query_pos.shape[0]
            perm = torch.randperm(nq, generator=perm_generator)
            query_pos = query_pos[[perm]]
            perm_inv = torch.argsort(perm)

        if query_pos.shape[0] > max_cond_samples:
            query_vals_init = self.model.sample_conditional(sample_val, sample_pos, query_pos[0:max_cond_samples], tau=tau)
            sample_val = torch.cat((sample_val, query_vals_init), dim=0)
            sample_pos = torch.cat((sample_pos,  query_pos[0:max_cond_samples]), dim=0)
            query_vals_end = self.model(sample_val, sample_pos, query_pos[max_cond_samples:]).mean()
            query_vals = torch.cat((query_vals_init, query_vals_end), dim=0)
        else:
            query_vals = self.model.sample_conditional(sample_val, sample_pos, query_pos, tau=tau)
        if permute_order:
            query_vals = query_vals[[perm_inv]]

        if permute_order and return_perm:
            return query_vals, perm
        else:
            return query_vals


class BaseTester(object):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.init_dataloader()

    def init_dataloader(self):
        cfg = self.cfg
        dset, self.pred_vis_fn = dataset_manager.val_dataset(cfg.data)
        self.dl = torch.utils.data.DataLoader(dset, batch_size=cfg.data.bs_val, num_workers=cfg.resources.num_workers)

    def extract_batch_input(self, batch, ns=None):
        sample_pos = batch['sample_positions'].permute(1,0,2).contiguous()
        sample_val = batch['sample_values'].permute(1,0,2).contiguous()
        if ns is not None:
            sample_pos = sample_pos[0:ns]
            sample_val = sample_val[0:ns]
        query_pos = batch['query_positions'].permute(1,0,2).contiguous()
        query_val = batch['query_values'].permute(1,0,2).contiguous()
        return sample_pos, sample_val, query_pos, query_val

    def test_dirs_init(self, eval_dir, vis_dir):
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        for f in glob.glob(vis_dir + '/*'):
            os.remove(f)

    def compute_sample_sizes(self):
        sample_sizes = []
        if self.cfg.eval.ns_steps > 1:
            log_step = np.log(self.cfg.eval.ns_max/self.cfg.eval.ns_min)/(self.cfg.eval.ns_steps-1)
            for sx in range(self.cfg.eval.ns_steps):
                ns = int(np.ceil(self.cfg.eval.ns_min*np.exp(log_step*sx)))
                sample_sizes.append(ns)
        else:
            sample_sizes = [self.cfg.eval.ns_min]
        return sample_sizes

    def test(self, model, eval_dir, **kwargs):
        pass



def init_model(cfg, _base_path):
    s2s_module = S2SModel(cfg=cfg)

    checkpoint_path = config.extract_ckpt_path(cfg.eval.checkpoint)
    checkpoint_path = osp.join(_base_path, cfg.logging.log_dir, checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    s2s_module.load_state_dict(checkpoint['state_dict'])
    s2s_module = s2s_module.cuda().eval()

    return s2s_module
