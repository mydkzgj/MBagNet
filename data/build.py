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
    labels = torch.tensor(labels, dtype=torch.int64)   # for continuous var CJY at 2020.9.5
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0), labels, imgs_path

def collate_fn(batch):
    if len(batch[0]) == 2:
        imgs, labels, = zip(*batch)
        imgs_path = [""] * len(batch)
    elif len(batch[0]) == 3:
        imgs, labels, imgs_path, = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.int64)   # for continuous var CJY at 2020.9.5
    return torch.stack(imgs, dim=0), labels, imgs_path

"""
def make_data_loader_for_classic_datasets(cfg, for_train):
    train_transforms = build_transforms(cfg, is_train=for_train)
    val_transforms = build_transforms(cfg, is_train=False)
    test_transforms = build_transforms(cfg, is_train=False)

    root_path = cfg.DATA.DATASETS.ROOT_DIR

    # for those have been realized by torchvision
    if cfg.DATA.DATASETS.NAMES == "cifar10":
        root_path = os.path.join(root_path, "DATABASE", "CIFAR10")
        train_set = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=train_transforms)
        val_set = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
    elif cfg.DATA.DATASETS.NAMES == "cifar100":
        root_path = os.path.join(root_path, "DATABASE", "CIFAR100")
        train_set = torchvision.datasets.CIFAR100(root=root_path, train=True, download=True, transform=train_transforms)
        val_set = torchvision.datasets.CIFAR100(root=root_path, train=False, download=True, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-classification":
        root_path = os.path.join(root_path, "DATABASE", "PASCAL-VOC")
        train_set = VOCClassification(root=root_path, year="2012", image_set="train", download=False, transform=train_transforms)
        val_set = VOCClassification(root=root_path, year="2012", image_set="val", download=False, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-detection":
        root_path = os.path.join(root_path, "DATABASE", "PASCAL-VOC")
        train_set = torchvision.datasets.VOCDetection(root=root_path, year="2012", image_set="train", download=True, transform=train_transforms)
        val_set = torchvision.datasets.VOCDetection(root=root_path, year="2012", image_set="val", download=True, transform=val_transforms)
        test_set = val_set
        classes_list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-segmentation":
        root_path = os.path.join(root_path, "DATABASE", "PASCAL-VOC")
        train_set = torchvision.datasets.VOCSegmentation(root=root_path, year="2012", image_set="train", download=True, transform=train_transforms)
        val_set = torchvision.datasets.VOCSegmentation(root=root_path, year="2012", image_set="val", download=True, transform=val_transforms)
        test_set = val_set
        classes_list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    elif cfg.DATA.DATASETS.NAMES == "coco-classification":
        root_path = os.path.join(root_path, "DATABASE", "Microsoft-COCO")
        year = "2017"
        annotation_path = os.path.join(root_path, "annotations")
        train_image_path = os.path.join(root_path, "{}{}".format("train", year))
        train_annfile =os.path.join(annotation_path, "instances_{}2017.json".format("train"))
        train_set = CocoClassification(root=train_image_path, annFile=train_annfile, transform=train_transforms)
        val_image_path = os.path.join(root_path, "{}{}".format("val", year))
        val_annfile = os.path.join(annotation_path, "instances_{}2017.json".format("val"))
        val_set = CocoClassification(root=val_image_path, annFile=val_annfile, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.class_list
    elif cfg.DATA.DATASETS.NAMES == "coco-caption":
        root_path = os.path.join(root_path, "DATABASE", "Microsoft-COCO")
        year = "2017"
        annotation_path = os.path.join(root_path, "annotations")
        train_image_path = os.path.join(root_path, "{}{}".format("train", year))
        train_annfile =os.path.join(annotation_path, "captions_{}2017.json".format("train"))
        train_set = torchvision.datasets.CocoCaptions(root=train_image_path, annFile=train_annfile, transform=train_transforms)
        val_image_path = os.path.join(root_path, "{}{}".format("val", year))
        val_annfile = os.path.join(annotation_path, "captions_{}2017.json".format("val"))
        val_set = torchvision.datasets.CocoCaptions(root=val_image_path, annFile=val_annfile, transform=val_transforms)
        test_set = val_set
        classes_list = None
    elif cfg.DATA.DATASETS.NAMES == "coco-detection":
        root_path = os.path.join(root_path, "DATABASE", "Microsoft-COCO")
        year = "2017"
        annotation_path = os.path.join(root_path, "annotations")
        train_image_path = os.path.join(root_path, "{}{}".format("train", year))
        train_annfile =os.path.join(annotation_path, "captions_{}{}.json".format("train", year))
        train_set = torchvision.datasets.CocoCaptions(root=train_image_path, annFile=train_annfile, transform=train_transforms)
        val_image_path = os.path.join(root_path, "{}{}".format("val", year))
        val_annfile = os.path.join(annotation_path, "captions_{}{}.json".format("val", year))
        val_set = torchvision.datasets.CocoCaptions(root=val_image_path, annFile=val_annfile, transform=val_transforms)
        test_set = val_set
        classes_list = [train_set.coco.dataset["categories"][i]["name"] for i in range(len(train_set.coco.dataset["categories"]))]

    return train_set, val_set, test_set, classes_list
"""

def make_data_loader(cfg, for_train):
    if cfg.DATA.DATASETS.NAMES == "none" or cfg.TRAIN.DATALOADER.IMS_PER_BATCH == 0:  #如果batch为0，那么就返回空  CJY at 2020.7.12
        classes_list = ["{}".format(i) for i in range(cfg.MODEL.CLA_NUM_CLASSES)]
        return None, None, None, classes_list

    # torchvision数据集（部分修改）
    torchvision_dataset_list = ["cifar10", "cifar100",
                                "pascal-voc-classification", "pascal-voc-detection", "pascal-voc-segmentation",
                                "coco-classification"]
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

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    num_classes = len(classes_list)

    train_sampler = build_sampler(cfg, train_set, num_classes, set_name="train", is_train=for_train)
    val_sampler = build_sampler(cfg, val_set, num_classes, set_name="val", is_train=False)
    test_sampler = build_sampler(cfg, test_set, num_classes, set_name="test", is_train=False)

    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH,
        sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.IMS_PER_BATCH, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.IMS_PER_BATCH, shuffle=False,
        sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn
    )
    #notes:
    #1.collate_fn是自定义函数，对提取的batch做处理，例如分开image和label

    """
    #train set # ？？是否要进行label-smoothing    
    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH,
        sampler=ClassBalanceRandomSampler(dataset.train, cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH,
                                          cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=for_train),
        num_workers=num_workers, collate_fn=collate_fn
    )

    #val set    
    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        #CJY at 2019.9.26 为了能够平衡样本
        sampler=ClassBalanceRandomSampler(dataset.val, cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH,
                                          cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=False),
        collate_fn=collate_fn
    )    
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        # CJY at 2019.9.26 为了能够平衡样本
        sampler=ClassBalanceRandomSampler(dataset.test, cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH,
                                          cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=False),
        collate_fn=collate_fn
    )
    #"""

    return train_loader, val_loader, test_loader, classes_list


#CJY at 2020.1.8  用于弱监督，引入segmentation_loader
def make_seg_data_loader(cfg, for_train):
    if cfg.DATA.DATASETS.SEG_NAMES == "none" or cfg.TRAIN.DATALOADER.MASK_PER_BATCH == 0:  #如果batch为0，那么就返回空  CJY at 2020.7.12
        classes_list = []
        return None, None, None, classes_list

    train_set, val_set, test_set, classes_list = make_seg_dataset_for_custom_datasets(cfg, for_train)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    num_classes = len(classes_list)

    train_sampler = build_seg_sampler(cfg, train_set, num_classes, set_name="train", is_train=for_train)
    val_sampler = build_seg_sampler(cfg, val_set, num_classes, set_name="val", is_train=False)
    test_sampler = build_seg_sampler(cfg, test_set, num_classes, set_name="test", is_train=False)

    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.MASK_PER_BATCH,
        sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn_seg
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.MASK_PER_BATCH, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, collate_fn=collate_fn_seg
    )

    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.MASK_PER_BATCH, shuffle=False,
        sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn_seg
    )

    """
    # train set
    # 是否要进行label-smoothing
    train_set = SegmentationDataset(dataset.seg_train, train_transforms, train_mask_transforms, cfg, is_train=for_train)
    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.MASK_PER_BATCH, num_workers=num_workers, #shuffle=True,
        # CJY  为了保证类别均衡
        sampler=ClassBalanceRandomSamplerForSegmentation(dataset.seg_train, 1, 1, num_classes, is_train=for_train),    #此处让所有分类标签的按顺序以1为单位交替进行
        collate_fn=collate_fn_seg
    )

    # val set
    val_set = SegmentationDataset(dataset.seg_val, val_transforms, val_mask_transforms, cfg, is_train=False)
    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.MASK_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn_seg
    )

    # test_set
    test_set = SegmentationDataset(dataset.seg_test, test_transforms, test_mask_transforms, cfg, is_train=False)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.MASK_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn_seg
    )
    #"""
    return train_loader, val_loader, test_loader, classes_list
