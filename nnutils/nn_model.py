from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pdb
import math
import torch
import torch.nn as nn

from pytorch3d.ops import knn


class NNNet(nn.Module):
    def __init__(self):
        super(NNNet, self).__init__()


    def forward(self, sample_vals, sample_posns, query_posns):
        '''
        Args:
            sample_vals: nS X B X ndv
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
        Returns:
            vals: nQ X B X ndv
        '''
        sample_vals = sample_vals.permute(1,0,2).contiguous()
        sample_posns = sample_posns.permute(1,0,2).contiguous()
        query_posns = query_posns.permute(1,0,2).contiguous()
        _, idx, _ = knn.knn_points(query_posns, sample_posns)
        vals = knn.knn_gather(sample_vals, idx)
        vals = vals[:,:,0,:].permute(1,0,2).contiguous()
        return vals

    def id_splat(self, sample_posns, query_posns, query_vals):
        '''
        Args:
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
            query_vals: nQ X B X ndv
        Returns:
            vals: nQ X B X ndv, with 0s everywhere but 1s at NNs of sample_posns
        '''
        ndv = query_vals.shape[-1]
        query_vals = query_vals.permute(1,0,2).contiguous()
        sample_posns = sample_posns.permute(1,0,2).contiguous()
        query_posns = query_posns.permute(1,0,2).contiguous()
        # idx is B X nS X 1
        _, idx, _ = knn.knn_points(sample_posns, query_posns)
        vals = query_vals[...,[0]]*0
        ones_s = torch.ones_like(sample_posns[...,[0]])
        vals.scatter_add_(dim=1, index=idx, src=ones_s)
        vals = torch.clamp(vals.repeat(1,1,ndv),0,1)
        vals = vals.permute(1,0,2).contiguous()
        return vals
        
