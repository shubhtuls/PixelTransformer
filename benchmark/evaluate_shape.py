'''
python -m s2s.benchmark.evaluate_shape data=shapenet eval.checkpoint.name=snet_chair data.ns=4096 data.nq_val=32768 eval.n_iter=10 eval.ns_min=32 eval.ns_max=32 eval.checkpoint.epoch=last data.nz_slices=32 eval.checkpoint.version=2 eval.ns_steps=1 eval.n_cond_samples=1 data.bs_val=1
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import glob
import math

_curr_path = osp.dirname(osp.abspath(__file__))
_base_path = osp.join(_curr_path, '..')

from omegaconf import DictConfig
import hydra
import pdb
import imageio
import numpy as np

import torch
import pytorch3d.structures
import pytorch3d.io as pio


from ..conf import config
from ..nnutils import renderer as render_util
from ..data import base_3d
from . import base as base_eval


class ShapeTester(base_eval.BaseTester):
    def __init__(self, cfg: DictConfig):
        super(ShapeTester, self).__init__(cfg)
        self._renderer = None

    def render_sdf(self, sdf):
        '''
        Args:
            sdf: B X 1 X D X H X W
        Returns:
            img: B X H_img X w_img X 3
        '''
        device = sdf.device
        if self._renderer is None:
            self._renderer = render_util.Pytorch3dRenderer(img_size=256, device=device)
            self._lights = render_util.ambient_light(device=device)
            self._cameras = render_util.stq_to_orthographic(render_util.canonical_pose())
            self._cameras = self._cameras.to(device=device)

        meshes = base_3d.extract_meshes(sdf)
        imgs = self._renderer.render_img(meshes, cameras=self._cameras, lights=self._lights, offset_z=1.)
        imgs = torch.flip(imgs, [1,2])
        return imgs[...,0:3]


    def pred_vis_nn(self, gt, sample_val, sample_pos):
        ns = gt.shape[0]
        bs = gt.shape[1]
        pred = gt.permute((1,0,2)).contiguous()
        
        D = self.cfg.data.nz_slices
        H = int(math.sqrt(ns//D))
        W = ns // (H*D)

        gt = gt.view(bs, 1, D, H, W)
        meshes_gt = base_3d.extract_meshes(gt)
        meshes_nn = base_3d.sdf_samples_mesh(0.5*sample_pos[:,:,[2,1,0]], sample_val)
        mesh_comb = pytorch3d.structures.join_meshes_as_scene([meshes_nn, meshes_gt])

        imgs = self._renderer.render_img(mesh_comb, cameras=self._cameras, lights=self._lights, offset_z=1.)
        imgs = torch.flip(imgs, [1,2])
        imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3])
        return imgs[...,0:3].detach().cpu().numpy(), mesh_comb


    def pred_to_mesh(self, pred):
        ns = pred.shape[0]
        bs = pred.shape[1]
        pred = pred.permute((1,0,2)).contiguous()
        
        D = self.cfg.data.nz_slices
        H = int(math.sqrt(ns//D))
        W = ns // (H*D)

        pred = pred.view(bs, 1, D, H, W)
        meshes = base_3d.extract_meshes(pred)
        return meshes


    def pred_vis_fn_3d(self, pred):
        ns = pred.shape[0]
        bs = pred.shape[1]
        pred = pred.permute((1,0,2)).contiguous()
        
        D = self.cfg.data.nz_slices
        H = int(math.sqrt(ns//D))
        W = ns // (H*D)

        pred = pred.view(bs, 1, D, H, W)
        imgs = self.render_sdf(pred)
        imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3])
        return imgs.detach().cpu().numpy()


    def test(self, model, eval_dir, nn_vis=False):
        vis_dir = osp.join(eval_dir, 'visualization')

        self.test_dirs_init(eval_dir, vis_dir)
        sample_sizes = self.compute_sample_sizes()

        for bx, batch in enumerate(self.dl):
            if bx > self.cfg.eval.n_iter:
                break

            sample_pos, sample_val, query_pos, query_val = self.extract_batch_input(batch)
            gt_img = (self.pred_vis_fn_3d(query_val.cuda())*255).astype(np.uint8)
            imageio.imsave(osp.join(vis_dir, 'iter_{}_gt.png'.format(bx)), gt_img)

            for sx, ns in enumerate(sample_sizes):
                with torch.no_grad():
                    pred = model(sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())

                pred_img = (self.pred_vis_fn_3d(pred)*255).astype(np.uint8)
                imageio.imsave(osp.join(vis_dir, 'iter_{}_pred_{}.png'.format(bx, ns)), pred_img)

                for rx in range(self.cfg.eval.n_cond_samples):
                    with torch.no_grad():
                        pred = model.forward_conditional(
                            sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())
                    pred_img = (self.pred_vis_fn_3d(pred)*255).astype(np.uint8)
                    pred_mesh = self.pred_to_mesh(pred)
                    pio.save_obj(osp.join(vis_dir, 'iter_{}_pred_{}_nc_{}.obj'.format(bx, ns, rx)),
                                 pred_mesh.verts_list()[0], pred_mesh.faces_list()[0])
                    imageio.imsave(osp.join(vis_dir, 'iter_{}_pred_{}_nc_{}.png'.format(bx, ns, rx)), pred_img)

                if nn_vis:
                    pred_img_nn, pred_mesh_nn = self.pred_vis_nn(
                        query_val.cuda(), sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns])
                    pred_img_nn = (pred_img_nn*255).astype(np.uint8)
                    imageio.imsave(osp.join(vis_dir, 'iter_{}_pred_nn_{}.png'.format(bx, ns)), pred_img_nn)
                    pio.save_obj(
                        osp.join(vis_dir, 'iter_{}_pred_nn_{}.obj'.format(bx, ns)),
                        pred_mesh_nn.verts_list()[0], pred_mesh_nn.faces_list()[0]
                    )


@hydra.main(config_name="config")
def main(cfg: config.Config) -> None:

    s2s_module = base_eval.init_model(cfg, _base_path)
    eval_dir = osp.join(_base_path, cfg.eval.eval_dir, cfg.eval.checkpoint.name, 'version_{}'.format(cfg.eval.checkpoint.version))

    torch.manual_seed(0)
    tester = ShapeTester(cfg)
    tester.test(s2s_module, eval_dir, nn_vis=True)


if __name__ == '__main__':
    main()