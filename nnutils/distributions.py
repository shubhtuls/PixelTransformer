'''
Define various output distributions that can be predicted
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb


import numpy as np
import torch
from torch.nn.functional import softmax
from pytorch3d.ops import knn_points
import torch.nn as nn


def pde_router(pde, ndv, bins_file=None):
    '''
    Args:
        pde: string indicating pde type
        ndv: dim of values
        bins_file: npy file with cluster centres
    Returns:
        out_pde_fn, ndv_out
    '''
    if pde == 'id':
        ndv_out = ndv
        out_pde_fn = Identity
    elif pde == 'std':
        ndv_out = ndv + 1
        out_pde_fn = StdGaussian
    elif pde == 'discrete':
        _centres = torch.Tensor(np.load(bins_file))
        ndv_out = _centres.shape[0]
        out_pde_fn = lambda params : DiscreteDistribution(params=params, centres=_centres)
    elif pde == 'disccont':
        _centres = torch.Tensor(np.load(bins_file))
        ndv_out = _centres.shape[0]*(ndv+2)
        # using a weight of 1 leads to very blurry output as it results in higher entropy cls predictions
        out_pde_fn = DiscreteContinuousDistributionModule(_centres, nll_reg_wt=0.1)
        # out_pde_fn = lambda params : DiscreteContinuousDistribution(params=params, centres=_centres)
    else:
        raise NotImplementedError

    return out_pde_fn, ndv_out
    


def nll_classification(query, bins, probs):
    '''
    Args:
        query: ... X ndv
        bins: nc X ndv
        probs: ... X nc
    Returns:
        nll: ... X 1 neg. log-likelihood
    '''
    nc, ndv = bins.shape
    query_shape = query.shape
    query = query.view(1,-1,ndv)

    bins = bins.view(1,-1,ndv)
    _, inds, _ = knn_points(query, bins, K=1)
    inds = inds.view(query_shape[:-1] + torch.Size([1]))
    probs = torch.gather(probs, -1, inds)
    return -1*torch.log(probs)


def identity(v):
    return v

def _repeat_last(v, k):
    return torch.cat(k*[v], dim=-1)

def discretize(query, bins):
    '''
    Args:
        query: ... X ndv
        bins: nc X ndv
    Returns:
        query_binned: ... X ndv
        inds: ... X 1
    '''
    bins = bins.to(query.device)
    nc, ndv = bins.shape
    query_shape = query.shape
    query = query.view(1,-1,ndv)
    bins = bins.view(1,-1,ndv)
    _, inds, _ = knn_points(query, bins, K=1)

    inds = inds.view(query_shape[:-1])
    vals = []
    for d in range(bins.shape[-1]):
        vals.append(torch.take(bins[0,:,d], inds))
    return torch.stack(vals, dim=-1), inds.unsqueeze(-1)



class BaseDistribution(object):
    def __init__(self, params=None):
        '''
        Args:
            params: output tensor dimension
        '''
        self._params = params

    def sample(self, **kwargs):
        '''
        Returns:
            samples: [...] X ndv, where params is of size [...] X nd_dist
        '''
        raise NotImplementedError
    
    def mle(self):
        '''
        Returns:
            samples: [...] X ndv, where params is of size [...] X nd_dist
        '''
        raise NotImplementedError
    
    def mean(self):
        '''
        Returns:
            samples: [...] X ndv, where params is of size [...] X nd_dist
        '''
        raise NotImplementedError

    def nll(self, query):
        '''
        Returns negative log likelihood of query (possibly unnormalized)
        Args:
            query: [...] X ndv, where params is of size [...] X nd_dist
        Returns:
            query_nll: [...] X 1
        '''
        raise NotImplementedError


class Identity(BaseDistribution):
    def mle(self):
        return self._params.clone()

    def sample(self, **kwargs):
        return self._params.clone()
    
    def mean(self):
        return self._params.clone()

    def nll(self, query):
        diff = self._params - query
        return torch.sum(torch.pow(diff, 2), dim=-1, keepdim=True)


class StdGaussian(BaseDistribution):
    def __init__(self, params):
        self._ndv = params.shape[-1] - 1
        self._mu = params[...,:-1]
        self._log_var = params[...,[-1]]
        
    def mle(self):
        return self._mu.clone()

    def mean(self):
        return self._mu.clone()

    def nll(self, query):
        diff_sq_sum = torch.sum(torch.pow(self._mu - query, 2), dim=-1, keepdim=True)
        return self._ndv*self._log_var + diff_sq_sum*torch.exp(-1*self._log_var)

    def sample(self, tau=1.):
        std = torch.exp(self._log_var/2)
        return torch.randn_like(self._mu)*std*tau + self._mu


class DiagonalGaussian(BaseDistribution):
    def __init__(self, params):
        self._ndv = int(params.shape[-1]//2)
        self._mu = params[...,:self._ndv]
        self._log_var = params[...,self._ndv:]

    def mle(self):
        return self._mu.clone()

    def mean(self):
        return self._mu.clone()

    def nll(self, query):
        diff_sq = torch.pow(self._mu - query, 2)*torch.exp(-1*self._log_var)
        return torch.sum(self._log_var + diff_sq, dim=-1, keepdim=True)

    def sample(self, tau=1.):
        std = torch.exp(self._log_var/2)
        return torch.randn_like(self._mu)*std*tau + self._mu


class DiscreteDistribution(object):
    def __init__(self, params=None, centres=None):
        self._centres = centres.to(params.device)
        self._nc, self._ndv = centres.shape
        self._probs = softmax(params, dim=-1)

    def mle(self):
        inds = torch.argmax(self._probs,dim=-1)
        vals = []
        for d in range(self._centres.shape[-1]):
            vals.append(torch.take(self._centres[:,d], inds))
        return torch.stack(vals, dim=-1)

    def mean(self):
        probs = self._probs.view(-1, 1, self._nc)
        centres = self._centres.permute(1,0).contiguous().view(1,-1,self._nc)
        vals = torch.sum(probs*centres, dim=-1)
        vals = vals.view(self._probs.shape[:-1] + torch.Size([self._ndv]))
        return vals
    
    def sample(self, tau=1.):
        probs = self._probs.view(-1, self._nc)
        probs = softmax(torch.log(probs)/tau, dim=-1)
        vals = []
        inds = torch.multinomial(probs, 1).view(self._probs.shape[:-1])
        for d in range(self._centres.shape[-1]):
            vals.append(torch.take(self._centres[:,d], inds))
        return torch.stack(vals, dim=-1)
        
    def nll(self, query):
        return nll_classification(query, self._centres, self._probs)

    def discretize(self, query):
        vals, _ = discretize(query, self._centres)
        return vals


class DiscreteContinuousDistributionModule(nn.Module):
    def __init__(self, centres, nll_reg_wt=0.1):
        super(DiscreteContinuousDistributionModule, self).__init__()
        self.nll_reg_wt = nll_reg_wt
        self.centres = centres

    def forward(self, params):
        return DiscreteContinuousDistribution(params, self.centres, nll_reg_wt=self.nll_reg_wt)


class DiscreteContinuousDistribution(object):
    def __init__(self, params=None, centres=None, nll_reg_wt=0.1):
        self._centres = centres.to(params.device)
        self._nc, self._ndv = centres.shape
        self.nll_reg_wt = nll_reg_wt
        # probs, logvar: ... X nc, mu: ... X nc X ndv
        self._probs, self._mu, self._log_var = torch.split(params, (self._nc, self._nc*self._ndv, self._nc), dim=-1)
        self._mu = self._mu.view(params.shape[:-1] + torch.Size([self._nc, self._ndv]))
        self._probs = softmax(self._probs, dim=-1)

        
    def nll(self, query):
        '''
        Args:
            query: ... X ndv
        Returns:
            log_probs: ... X 1
        '''
        nll_cls = nll_classification(query, self._centres, self._probs)

        query_centres, bin_inds = discretize(query, self._centres) # ... X ndv, ... X 1
        query_diffs = (query - query_centres) # ... X ndv
        mu_bins = torch.gather(self._mu, -2, _repeat_last(bin_inds.unsqueeze(-1), self._ndv)).squeeze(dim=-2) # ... X ndv
        log_var_bins = torch.gather(self._log_var, -1, bin_inds) # ... X 1
        diff_sq = torch.pow(mu_bins - query_diffs, 2)*torch.exp(-1*log_var_bins)
        nll_reg = torch.sum(log_var_bins + diff_sq, -1, keepdim=True)
        return nll_cls + self.nll_reg_wt*nll_reg

    def sample(self, tau=1.):
        '''
        tau: temperature. 0-> MLE, inf-> random, 1-> normal sampling
        '''
        probs = self._probs.view(-1, self._nc)
        probs = softmax(torch.log(probs)/tau, dim=-1)

        vals = []
        inds = torch.multinomial(probs, 1).view(self._probs.shape[:-1])
        for d in range(self._centres.shape[-1]):
            vals.append(torch.take(self._centres[:,d], inds))
        vals = torch.stack(vals, dim=-1)

        ## add gaussian term's effect
        inds = inds.unsqueeze(-1)
        mu = torch.gather(self._mu, -2,  _repeat_last(inds.unsqueeze(-1), self._ndv)).squeeze(dim=-2) # ... X ndv
        log_var = torch.gather(self._log_var, -1, inds) # ... X 1
        std = torch.exp(log_var)
        return torch.randn_like(mu)*std*tau + mu + vals
        # return mu + vals

    def mle(self):
        inds = torch.argmax(self._probs,dim=-1)
        vals = []
        for d in range(self._centres.shape[-1]):
            vals.append(torch.take(self._centres[:,d], inds))
        vals = torch.stack(vals, dim=-1)

        ## add gaussian term's effect
        inds = inds.unsqueeze(-1)
        mu = torch.gather(self._mu, -2, _repeat_last(inds.unsqueeze(-1), self._ndv)).squeeze(dim=-2) # ... X ndv
        return vals + mu

    def mean(self):
        probs = self._probs.view(-1, self._nc, 1) # nQ X nc X 1
        centres = self._centres.view(1,self._nc,-1) # 1 X nc X ndv
        deltas = self._mu.view(-1, self._nc, self._ndv) # nQ X nc X ndv

        vals = torch.sum(probs*(centres+deltas), dim=-2)
        vals = vals.view(self._probs.shape[:-1] + torch.Size([self._ndv]))
        return vals
