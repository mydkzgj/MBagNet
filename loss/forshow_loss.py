# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F

class ForShowLoss(object):
    def __init__(self):
        pass

    def __call__(self, show):
        total_loss = show
        return total_loss

