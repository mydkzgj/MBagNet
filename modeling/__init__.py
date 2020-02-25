# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline

def build_model(cfg, num_classes):
    model = Baseline(cfg.MODEL.NAME, num_classes,
                     preAct=cfg.MODEL.PRE_ACTIVATION, fusionType=cfg.MODEL.FUSION_TYPE,
                     base_classifier_Type=cfg.MODEL.BASE_CLASSIFIER_COMBINE_TYPE,
                     hookType=cfg.MODEL.HOOK_TYPE, segmentationType=cfg.MODEL.SEGMENTATION_TYPE, seg_num_classes=cfg.MODEL.SEG_NUM_CLASSES,
                     accumulation_steps = cfg.TRAIN.DATALOADER.ACCUMULATION_STEP
                     )
    return model
