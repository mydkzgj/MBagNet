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
        super(AutoWeightedRandomSampler, self).__init__()
        self.weights = [0, 1]
        print(1)