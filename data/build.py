# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch

from .datasets import make_dataset_for_classic_datasets, make_dataset_for_custom_datasets, make_seg_dataset_for_custom_datasets

from .samplers import build_sampler, build_seg_sampler

from torch.utils.data import DataLoader


def collate_fn_seg(batch):
    if len(batch[0]) == 3:
        imgs, masks, labels, = zip(*batch)
        imgs_path = [""] * len(batch)
    elif len(batch[0]) == 4:
        imgs, masks, labels, imgs_path = zip(*batch)
    elif len(batch[0]) == 2:
        imgs, masks = zip(*batch)
        labels = None
        imgs_path = [""] * len(batch)
    labels = torch.tensor(labels, dtype=torch.int64)   # for continuous var CJY at 2020.9.5
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0), labels, imgs_path

def collate_fn(batch):
    if len(batch[0]) == 2:
        imgs, labels, = zip(*batch)
        imgs_path = [""] * len(batch)
    elif len(batch[0]) == 3:
        imgs, labels, imgs_path, = zip(*batch)
    if isinstance(labels[0], dict) == False:
        labels = torch.tensor(labels, dtype=torch.int64)   # for continuous var CJY at 2020.9.5
    return torch.stack(imgs, dim=0), labels, imgs_path


def make_data_loader(cfg, for_train):
    if cfg.DATA.DATASETS.NAMES == "none" or cfg.TRAIN.DATALOADER.IMS_PER_BATCH == 0:  #如果batch为0，那么就返回空  CJY at 2020.7.12
        classes_list = ["{}".format(i) for i in range(cfg.MODEL.CLA_NUM_CLASSES)]
        return None, None, None, classes_list

    # torchvision数据集（部分修改）
    torchvision_dataset_list = ["cifar10", "cifar100",
                                "pascal-voc-classification", "pascal-voc-detection", "pascal-voc-segmentation",
                                "coco-classification", "coco-detection"]
    # 自定义数据集
    custom_dataset_list = ["ddr_dr_grading", "ddr_lesion_segmentation_regroup",
                           "ddr_lesion_segmentation_multilabel_weaksupervision",
                           "ddr_lesion_segmentation_multilabel_weaksupervision_colormask",
                           "fundusTR", "fundusTRjoint",
                           "examples"]

    if cfg.DATA.DATASETS.NAMES in torchvision_dataset_list:
        train_set, val_set, test_set, classes_list = make_dataset_for_classic_datasets(cfg, for_train)
    elif cfg.DATA.DATASETS.NAMES in custom_dataset_list:
        train_set, val_set, test_set, classes_list = make_dataset_for_custom_datasets(cfg, for_train)

    num_classes = len(classes_list)
    train_sampler = build_sampler(cfg, train_set, num_classes, set_name="train", is_train=for_train)
    val_sampler = build_sampler(cfg, val_set, num_classes, set_name="val", is_train=False)
    test_sampler = build_sampler(cfg, test_set, num_classes, set_name="test", is_train=False)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    if cfg.MODEL.CLASSIFIER_OUTPUT_TYPE == "multi-label":
        #multi - label状态时，metric的计算我取巧了，但是需要保证每个batch中的数据个数保持不变（last batch可能会少）
        drop_last = True
    else:
        drop_last = False

    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH, sampler=train_sampler,
         num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.IMS_PER_BATCH, sampler=val_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )

    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.IMS_PER_BATCH, sampler=test_sampler,
         num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )
    #notes:
    #1.collate_fn是自定义函数，对提取的batch做处理，例如分开image和label
    return train_loader, val_loader, test_loader, classes_list


#CJY at 2020.1.8  用于弱监督，引入segmentation_loader
def make_seg_data_loader(cfg, for_train):
    if cfg.DATA.DATASETS.SEG_NAMES == "none" or cfg.TRAIN.DATALOADER.MASK_PER_BATCH == 0:  #如果batch为0，那么就返回空  CJY at 2020.7.12
        classes_list = []
        return None, None, None, classes_list

    # torchvision分割数据集（部分修改）
    torchvision_dataset_list = ["pascal-voc-segmentation",
                                "coco-classification", "coco-detection"]
    # 自定义数据集
    custom_seg_dataset_list = ["ddr_lesion_segmentation_regroup",
                               "ddr_lesion_segmentation_multilabel_weaksupervision",
                               "ddr_lesion_segmentation_multilabel_weaksupervision_colormask",
                               ]

    if cfg.DATA.DATASETS.SEG_NAMES in torchvision_dataset_list:
        train_set, val_set, test_set, classes_list = make_dataset_for_classic_datasets(cfg, for_train, "segmentation")
    elif cfg.DATA.DATASETS.SEG_NAMES in custom_seg_dataset_list:
        train_set, val_set, test_set, classes_list = make_seg_dataset_for_custom_datasets(cfg, for_train)

    num_classes = len(classes_list)
    train_sampler = build_seg_sampler(cfg, train_set, num_classes, set_name="train", is_train=for_train)
    val_sampler = build_seg_sampler(cfg, val_set, num_classes, set_name="val", is_train=False)
    test_sampler = build_seg_sampler(cfg, test_set, num_classes, set_name="test", is_train=False)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    if cfg.MODEL.CLASSIFIER_OUTPUT_TYPE == "multi-label":
        # multi - label状态时，metric的计算我取巧了，但是需要保证每个batch中的数据个数保持不变（last batch可能会少）
        drop_last = True
    else:
        drop_last = False

    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.MASK_PER_BATCH, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn_seg, drop_last=drop_last
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.MASK_PER_BATCH, sampler=val_sampler,
        num_workers=num_workers, collate_fn=collate_fn_seg, drop_last=drop_last
    )

    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.MASK_PER_BATCH, sampler=test_sampler,
        num_workers=num_workers, collate_fn=collate_fn_seg, drop_last=drop_last
    )
    return train_loader, val_loader, test_loader, classes_list
