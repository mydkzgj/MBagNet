# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline

def build_model(cfg, num_classes):
    model = Baseline(cfg.MODEL.NAME, num_classes,)
    return model
