from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def subsampling_manager(ss_mode, max_ss):
    '''
    Args:
        pde: string indicating pde type
        ndv: dim of values
        bins_file: npy file with cluster centres
    Returns:
        out_pde_fn, ndv_out
    '''
    if ss_mode == 'linear':
        return LinearSampler(1./max_ss)
    elif ss_mode == 'log':
        return LogSampler(max_ss)
    elif ss_mode == 'loglinear':
        return LogLinearSampler(max_ss)
    else:
        raise NotImplementedError


class BaseSampler(object):
    def __init__(self, param):
        '''
        Args:
            params: output tensor dimension
        '''
        pass
    
    def sample(self):
        pass
    
    def median(self):
        pass


class LinearSampler(BaseSampler):
    def __init__(self, min_range):
        self._min = min_range

    def sample(self):
        return 1 - np.random.rand()*(1-self._min)
    
    def median(self):
        return  0.5*(1 + self._min)


class LogSampler(BaseSampler):
    def __init__(self, max_ratio):
        self.max_ratio = max_ratio

    def sample(self):
        return np.exp(-np.log(self.max_ratio)*np.random.rand())
    
    def median(self):
        return  np.exp(-np.log(self.max_ratio)*0.5)


class LogLinearSampler(BaseSampler):
    def __init__(self, max_ratio):
        self._s0 = LinearSampler(0)
        self._s1 = LogSampler(max_ratio)
    
    def sample(self):
        if np.random.rand() > 0.5:
            return self._s1.sample()
        else:
            return self._s0.sample()
    
    def median(self):
        return self._s1.median()