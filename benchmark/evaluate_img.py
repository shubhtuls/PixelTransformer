'''
python -m PixelTransformer.benchmark.evaluate_img data.bs_val=1 data=cats eval.checkpoint.name=cats data.ns=4096 data.nq_val=4096 eval.n_iter=32 eval.ns_min=32 eval.ns_max=32 eval.checkpoint.epoch=last eval.ns_steps=1 eval.n_cond_samples=1

python ~/scripts/cluster_run.py --setup=/private/home/shubhtuls/scripts/init_s2s_cluster.sh --partition='dev' --cmd=''
'''

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
from skimage.metrics import structural_similarity

import torch

from ..external.pytorch_cifar import models as cifar_models
from ..conf import config
from . import base as base_eval


class ImgTester(base_eval.BaseTester):
    def test(self, model, eval_dir, nn_vis=False, generation_vis=False):

        cfg = self.cfg
        if cfg.eval.classify:
            cls_net = cifar_models.ResNet18().cuda()
            cls_net = torch.nn.DataParallel(cls_net)
            checkpoint = torch.load(osp.join(_base_path, 'external', 'pytorch_cifar', 'checkpoint', 'ckpt.pth'))
            cls_net.load_state_dict(checkpoint['net'])
            cls_net.eval()

        vis_dir = osp.join(eval_dir, 'visualization')
        self.test_dirs_init(eval_dir, vis_dir)
        sample_sizes = self.compute_sample_sizes()

        metrics = {'ssim':[], 'ssim_sample':[]}
        if cfg.eval.classify:
            metrics['cls'] = []
            metrics['cls_gt'] = []
            metrics['cls_sample'] = []

        perm_generator = torch.Generator(device='cpu')
        perm_generator.manual_seed(0)

        for bx, batch in enumerate(self.dl):
            if bx >= self.cfg.eval.n_iter:
                break

            sample_pos, sample_val, query_pos, query_val = self.extract_batch_input(batch)
            gt_img = (self.pred_vis_fn(query_pos, query_val)*255).astype(np.uint8)
            imageio.imsave(osp.join(vis_dir, 'iter_{}_gt.png'.format(bx)), gt_img)

            for sx, ns in enumerate(sample_sizes):
                if nn_vis:
                    pred_nn = model.nn_model(sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())
                    pred_img_nn = (self.pred_vis_fn(query_pos, pred_nn)*255).astype(np.uint8)
                    imageio.imsave(osp.join(vis_dir, 'iter_{}_pred_nn_{}.png'.format(bx, ns)), pred_img_nn)

                with torch.no_grad():
                    pred = model(sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda())

                if cfg.eval.classify:
                    pred_im = pred.view(32,32,1,3).permute(2,3,0,1).contiguous()
                    gt_im = query_val.cuda().view(32,32,1,3).permute(2,3,0,1).contiguous()
                    pred_cls = cls_net(pred_im).argmax()
                    metrics['cls'].append((pred_cls == batch['label'].cuda()).sum().item())
                    metrics['cls_gt'].append((cls_net(gt_im).argmax() == batch['label'].cuda()).sum().item())
                    

                pred_img = (self.pred_vis_fn(query_pos, pred)*255).astype(np.uint8)
                imageio.imsave(osp.join(vis_dir, 'iter_{}_pred_{}.png'.format(bx, ns)), pred_img)
                metrics['ssim'].append(
                    structural_similarity(gt_img, pred_img, data_range=255, multichannel=True))

                for rx in range(self.cfg.eval.n_cond_samples):
                    with torch.no_grad():
                        pred, pred_seq_order = model.forward_conditional(
                            sample_val.cuda()[0:ns], sample_pos.cuda()[0:ns], query_pos.cuda(),
                            return_perm=True, perm_generator=perm_generator
                        )
                    pred_img = (self.pred_vis_fn(query_pos, pred)*255).astype(np.uint8)
                    imageio.imsave(osp.join(vis_dir, 'iter_{}_pred_{}_nc_{}.png'.format(bx, ns, rx)), pred_img)

                    if generation_vis:
                        init_mask = model.nn_model.id_splat(sample_pos.cuda()[0:ns], query_pos.cuda(), query_val.cuda())
                        pred_img_seq = []
                        for nq in range(pred.shape[0]):
                            if (nq % 100 == 0) or nq == (pred.shape[0]-1):
                                pred_img_seq.append(
                                    (self.pred_vis_fn(query_pos, pred*init_mask + (1-init_mask))*255).astype(np.uint8))
                            init_mask[pred_seq_order[nq],:,:] = 1
                        imageio.mimsave(osp.join(vis_dir, 'iter_{}_pred_{}_nc_{}.gif'.format(bx, ns, rx)), pred_img_seq)
                        
                    metrics['ssim_sample'].append(
                        structural_similarity(gt_img, pred_img, data_range=255, multichannel=True))
                    if cfg.eval.classify:
                        pred_im = pred.view(32,32,1,3).permute(2,3,0,1).contiguous()
                        pred_cls = cls_net(pred_im).argmax()
                        metrics['cls_sample'].append((pred_cls == batch['label'].cuda()).sum().item())

        np.save(osp.join(eval_dir, 'sample_sizes.npy'), np.array(sample_sizes))
        for k in metrics.keys():
            metrics[k] = np.array(metrics[k]).reshape((-1,self.cfg.eval.ns_steps))
            np.save(osp.join(eval_dir, '{}.npy'.format(k)), metrics[k])
            print(k, metrics[k].mean())


@hydra.main(config_name="config")
def main(cfg: config.Config) -> None:

    s2s_module = base_eval.init_model(cfg, _base_path)
    eval_dir = osp.join(_base_path, cfg.eval.eval_dir, cfg.eval.checkpoint.name, 'version_{}'.format(cfg.eval.checkpoint.version))

    torch.manual_seed(0)
    tester = ImgTester(cfg)
    tester.test(s2s_module, eval_dir, nn_vis=True)


if __name__ == '__main__':
    main()