# encoding: utf-8
"""
@author:  JiayangChen
@contact: sychenjiayang@163.com
"""

import random
from collections import defaultdict

from torch.utils.data.sampler import WeightedRandomSampler


#CJY at 2019.9.26
class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, data_source, replacement=True):
        weights = [0.2, 0.4, 0.8, 1]
        num_samples = 100
        super(AutoWeightedRandomSampler, self).__init__(weights, num_samples)
        print(1)