# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline

def build_model(cfg, num_classes):
    model = Baseline(cfg.MODEL.NAME, num_classes,
                     preAct=cfg.MODEL.PRE_ACTIVATION, fusionType=cfg.MODEL.FUSION_TYPE,
                     base_classifier_Type=cfg.MODEL.BASE_CLASSIFIER_COMBINE_TYPE, hierarchy_classifier=cfg.MODEL.HIRERARCHY_CLASSIFIER,
                     hookType=cfg.MODEL.HOOK_TYPE, segmentationType=cfg.MODEL.SEGMENTATION_TYPE, seg_num_classes=cfg.MODEL.SEG_NUM_CLASSES, segSupervisedType=cfg.MODEL.SEG_SUPERVISED_TYPE,
                     gcamSupervisedType=cfg.MODEL.GCAM_SUPERVISED_TYPE, guidedBP=cfg.MODEL.GCAM_GUIDED_BP,
                     maskedImgReloadType=cfg.MODEL.MASKED_IMG_RELOAD_TYPE, preReload=cfg.MODEL.PRE_RELOAD,
                     branch_img_num=cfg.MODEL.BRANCH_IMG_NUM, branchConfigType=cfg.MODEL.BRANCH_CONFIG_TYPE,
                     accumulation_steps=cfg.TRAIN.DATALOADER.ACCUMULATION_STEP
                     )
    return model
