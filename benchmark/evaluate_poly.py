'''
python -m s2s.benchmark.evaluate_poly data=polynomial data.degree=6 eval.checkpoint.name=poly_pf eval.n_iter=10 eval.ns_min=4 eval.ns_max=12 eval.ns_steps=3 eval.n_cond_samples=10 data.ns=30 model.out_pde=std
'''

import os
import os.path as osp
_curr_path = osp.dirname(osp.abspath(__file__))
_base_path = osp.join(_curr_path, '..')

from omegaconf import DictConfig
import hydra
import pdb
import imageio
import numpy as np

import torch

from ..conf import config
from ..nnutils import distributions as distribution_util
from . import base as base_eval


class PolyTester(base_eval.BaseTester):

    def test(self, model, eval_dir, nn_vis=False):
        vis_dir = osp.join(eval_dir, 'visualization')
        self.test_dirs_init(eval_dir, vis_dir)
        sample_sizes = self.compute_sample_sizes()
    
        for bx, batch in enumerate(self.dl):
            if bx > self.cfg.eval.n_iter:
                break

            sample_pos, sample_val, query_pos, query_val = self.extract_batch_input(batch)
            np.save(osp.join(eval_dir, 'iter_{:03d}_gt.npy'.format(bx)), query_val.numpy().reshape((-1)))

            for sx, ns in enumerate(sample_sizes):
                with torch.no_grad():
                    pred = model(sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())


                np.save(osp.join(eval_dir, 'iter_{:03d}_ns{:03d}_pred.npy'.format(bx, ns)), pred.detach().cpu().numpy().reshape((-1)))
                np.save(osp.join(eval_dir, 'iter_{:03d}_ns{:03d}_spos.npy'.format(bx, ns)), sample_pos[0:ns].numpy().reshape((-1)))
                np.save(osp.join(eval_dir, 'iter_{:03d}_ns{:03d}_sval.npy'.format(bx, ns)), sample_val[0:ns].numpy().reshape((-1)))

                for rx in range(self.cfg.eval.n_cond_samples):
                    pred = model.forward_conditional(
                        sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())
                    np.save(osp.join(eval_dir, 'iter_{:03d}_ns{:03d}_pred{:03d}.npy'.format(bx, ns, rx)), pred.detach().cpu().numpy().reshape((-1)))


@hydra.main(config_name="config")
def main(cfg: config.Config) -> None:
    s2s_module = base_eval.init_model(cfg, _base_path)
    eval_dir = osp.join(_base_path, cfg.eval.eval_dir, cfg.eval.checkpoint.name, 'version_{}'.format(cfg.eval.checkpoint.version))

    torch.manual_seed(0)
    tester = PolyTester(cfg)
    tester.test(s2s_module, eval_dir, nn_vis=True)


if __name__ == '__main__':
    main()