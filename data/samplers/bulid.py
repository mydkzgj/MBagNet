# encoding: utf-8
"""
@author:  chenjiayang
@contact: sychenjiayang@163.com
"""
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler
from .class_balance_random_sampler import ClassBalanceRandomSampler, ClassBalanceRandomSamplerForSegmentation
#from torch.utils.data.sampler import WeightedRandomSampler
from .weighted_random_sampler import AutoWeightedRandomSampler


def build_sampler(cfg, data_source, num_classes, set_name="train", is_train=True):
    if cfg.DATA.DATALOADER.SAMPLER == "sequential":
        sampler = SequentialSampler(data_source)
    elif cfg.DATA.DATALOADER.SAMPLER == "random":
        sampler = RandomSampler(data_source, replacement=False)
    elif cfg.DATA.DATALOADER.SAMPLER == "weighted_random":
        sampler = AutoWeightedRandomSampler(data_source, replacement=True)
    elif cfg.DATA.DATALOADER.SAMPLER == "class_balance_random":
        if set_name == "train":
            num_categories_per_batch = cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH
            num_instances_per_category = cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
        elif set_name == "val":
            num_categories_per_batch = cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH
            num_instances_per_category = cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
        elif set_name == "test":
            num_categories_per_batch = cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH
            num_instances_per_category = cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
        else:
            raise Exception("Wrong Set Name For Sampler!")
        max_num_categories = num_classes
        sampler = ClassBalanceRandomSampler(data_source, num_categories_per_batch, num_instances_per_category, max_num_categories, is_train=is_train)
    else:
        raise Exception("Wrong Sampler Name!")
    return sampler


def build_seg_sampler(cfg, data_source, num_classes, set_name="train", is_train=True):
    if cfg.DATA.DATALOADER.SAMPLER == "sequential":
        sampler = SequentialSampler(data_source)
    elif cfg.DATA.DATALOADER.SAMPLER == "random":
        sampler = ClassBalanceRandomSampler(data_source, replacement=False)
    elif cfg.DATA.DATALOADER.SAMPLER == "weighted_random":
        sampler = AutoWeightedRandomSampler(data_source, replacement=True)
    elif cfg.DATA.DATALOADER.SAMPLER == "class_balance_random":
        # 此处让所有分类标签的按顺序以1为单位交替进行
        if set_name == "train":
            num_categories_per_batch = 1
            num_instances_per_category = 1
        elif set_name == "val":
            num_categories_per_batch = 1
            num_instances_per_category = 1
        elif set_name == "test":
            num_categories_per_batch = 1
            num_instances_per_category = 1
        else:
            raise Exception("Wrong Set Name For Sampler!")
        max_num_categories = num_classes
        sampler = ClassBalanceRandomSamplerForSegmentation(data_source, num_categories_per_batch, num_instances_per_category, max_num_categories, is_train=is_train)
    else:
        raise Exception("Wrong Sampler Name!")
    return sampler


