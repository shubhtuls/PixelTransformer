'''
python -m s2s.benchmark.evaluate_video data=beach data.n_val_frames=17 eval.checkpoint.name=beach_pf1 eval.n_iter=16 eval.ns_min=1024 eval.ns_max=1024 eval.checkpoint.epoch=last data.nq_val=69632 eval.checkpoint.version=0 eval.ns_steps=1 eval.n_cond_samples=0 data.ns=4096

python ~/scripts/cluster_run.py --setup=/private/home/shubhtuls/scripts/init_s2s_cluster.sh --partition='dev' --cmd='' --max_mem
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import glob

_curr_path = osp.dirname(osp.abspath(__file__))
_base_path = osp.join(_curr_path, '..')

from omegaconf import DictConfig
import hydra
import pdb
import imageio
import numpy as np
import math

import torch

from ..conf import config
from . import base as base_eval


class VideoTester(base_eval.BaseTester):
    def test(self, model, eval_dir, nn_vis=False):
        vis_dir = osp.join(eval_dir, 'visualization')
        self.test_dirs_init(eval_dir, vis_dir)
        sample_sizes = self.compute_sample_sizes()

        n_val_frames = self.cfg.data.n_val_frames
        H = int(math.sqrt(self.cfg.data.nq_val // n_val_frames))

        for bx, batch in enumerate(self.dl):
            if bx >= self.cfg.eval.n_iter:
                break

            # pdb.set_trace()
            sample_pos, sample_val, query_pos, query_val = self.extract_batch_input(batch)
            gt_video = (query_val.view(n_val_frames, H, H, 3).cpu().detach().numpy()*255).astype(np.uint8)
            imageio.mimsave(osp.join(vis_dir, 'iter_{}_gt.gif'.format(bx)), gt_video)

            for sx, ns in enumerate(sample_sizes):
                with torch.no_grad():
                    pred = model(sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())

                pred_video = (pred.view(n_val_frames, H, H, 3).cpu().detach().numpy()*255).astype(np.uint8)
                imageio.mimsave(osp.join(vis_dir, 'iter_{}_pred_{}.gif'.format(bx, ns)), pred_video)

                if nn_vis:
                    pred_nn = model.nn_model(sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())
                    pred_video_nn = (pred_nn.view(n_val_frames, H, H, 3).cpu().detach().numpy()*255).astype(np.uint8)
                    imageio.mimsave(osp.join(vis_dir, 'iter_{}_pred_nn_{}.gif'.format(bx, ns)), pred_video_nn)

                    pred_nn_splat = model.nn_model.id_splat(sample_pos.cuda()[0:ns], query_pos.cuda(), query_val.cuda())
                    pred_nn_splat[...,1:3] *= 0

                    # red values at observed pixels
                    pred_n_samples = pred*(1-pred_nn_splat[...,[0]]) + pred_nn_splat
                    pred_video_nn = (pred_n_samples.view(n_val_frames, H, H, 3).cpu().detach().numpy()*255).astype(np.uint8)
                    imageio.mimsave(osp.join(vis_dir, 'iter_{}_pred_n_samples_{}.gif'.format(bx, ns)), pred_video_nn)

                for rx in range(self.cfg.eval.n_cond_samples):
                    with torch.no_grad():
                        pred = model.forward_conditional(
                            sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())
                    pred_video = (pred.view(n_val_frames, H, H, 3).cpu().detach().numpy()*255).astype(np.uint8)
                    imageio.mimsave(osp.join(vis_dir, 'iter_{}_pred_{}_nc_{}.gif'.format(bx, ns, rx)), pred_video)


@hydra.main(config_name="config")
def main(cfg: config.Config) -> None:

    s2s_module = base_eval.init_model(cfg, _base_path)
    eval_dir = osp.join(_base_path, cfg.eval.eval_dir, cfg.eval.checkpoint.name, 'version_{}'.format(cfg.eval.checkpoint.version))

    torch.manual_seed(0)
    tester = VideoTester(cfg)
    tester.test(s2s_module, eval_dir, nn_vis=True)


if __name__ == '__main__':
    main()