# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .datasets import init_dataset, ImageDataset, SegmentationDataset
from .samplers import RandomSampler, RandomSamplerForSegmentation
from .transforms import build_transforms, build_seg_transforms

import torch
from torch.utils.data import DataLoader
import torchvision

import os

from .datasets.lib.pascal_voc_classification import VOCClassification

def collate_fn_seg(batch):
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

def make_data_loader_for_classic_datasets(cfg, for_train):
    train_transforms = build_transforms(cfg, is_train=for_train)
    val_transforms = build_transforms(cfg, is_train=False)
    test_transforms = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    root_path = cfg.DATA.DATASETS.ROOT_DIR
    # for those have been realized by torchvision
    if cfg.DATA.DATASETS.NAMES == "cifar10":
        root_path = os.path.join(root_path, "CIFAR10")
        train_set = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=train_transforms)
        val_set = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
        num_classes = len(train_set.classes)
    elif cfg.DATA.DATASETS.NAMES == "cifar100":
        root_path = os.path.join(root_path, "CIFAR100")
        train_set = torchvision.datasets.CIFAR100(root=root_path, train=True, download=True, transform=train_transforms)
        val_set = torchvision.datasets.CIFAR100(root=root_path, train=False, download=True, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
        num_classes = len(train_set.classes)
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-classification":
        root_path = os.path.join(root_path, "PASCAL-VOC")
        train_set = VOCClassification(root=root_path, year="2012", image_set="train", download=True, transform=train_transforms)
        val_set = VOCClassification(root=root_path, year="2012", image_set="val", download=True, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
        num_classes = len(train_set.classes)
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-detection":
        root_path = os.path.join(root_path, "PASCAL-VOC")
        train_set = torchvision.datasets.VOCDetection(root=root_path, year="2012", image_set="train", download=True, transform=train_transforms)
        val_set = torchvision.datasets.VOCDetection(root=root_path, year="2012", image_set="val", download=True, transform=val_transforms)
        test_set = val_set
        classes_list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        num_classes = len(classes_list)
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-segmentation":
        root_path = os.path.join(root_path, "PASCAL-VOC")
        train_set = torchvision.datasets.VOCSegmentation(root=root_path, year="2012", image_set="train", download=True, transform=train_transforms)
        val_set = torchvision.datasets.VOCSegmentation(root=root_path, year="2012", image_set="val", download=True, transform=val_transforms)
        test_set = val_set
        classes_list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        num_classes = len(classes_list)

    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH,
        sampler=RandomSampler(train_set, cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH,
                              cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=for_train),
        num_workers=num_workers, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        # CJY at 2019.9.26 为了能够平衡样本
        sampler=RandomSampler(val_set, cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH,
                              cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=False),
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        # CJY at 2019.9.26 为了能够平衡样本
        sampler=RandomSampler(test_set, cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH,
                              cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=False),
        #collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, classes_list


    # CJY at 2019.11.20 加入其他非医学图像数据集
    """
    if cfg.DATA.DATASETS.NAMES in ["cifa10"]:
        BATCH_SIZE = 128  # 批处理尺寸(batch_size)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='D:/Research/DL/Database/CIFA10/', train=True, download=True,
                                                transform=transform_train)  # 训练数据集
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=1)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

        test_set = torchvision.datasets.CIFAR10(root='D:/Research/DL/Database/CIFA10/', train=False, download=True,
                                                transform=transform_test)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)
        # cifar-10的标签
        classes_list = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_loader, val_loader, test_loader, classes_list
    elif cfg.DATA.DATASETS.NAMES in ["imagenet"]:  # 好像有不同的预处理方式
        BATCH_SIZE = 64  # 批处理尺寸(batch_size)
        transform_train = transforms.Compose([  # transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        transform_test = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

        # Pass transforms in here, then run the next cell to see how the transforms look
        trainset = torchvision.datasets.ImageNet(root='D:/Research/DL/Database/ImageNet/Val', split="train",
                                                 download=True,
                                                 transform=transform_train)  # 训练数据集
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=1)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

        val_set = torchvision.datasets.ImageNet(root='D:/Research/DL/Database/ImageNet/Val', split="val", download=True,
                                                transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        # cifar-10的标签
        classes_list = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_loader, val_loader, test_loader, classes_list
    """


def make_data_loader(cfg, for_train):
    if cfg.DATA.DATASETS.NAMES == "none" or cfg.TRAIN.DATALOADER.IMS_PER_BATCH == 0:  #如果batch为0，那么就返回空  CJY at 2020.7.12
        classes_list = ["{}".format(i) for i in range(cfg.MODEL.CLA_NUM_CLASSES)]
        return None, None, None, classes_list

    torchvision_dataset_list = ["cifar10", "cifar100",
                                "pascal-voc-classification", "pascal-voc-detection", "pascal-voc-segmentation",
                                "coco"]
    #custom_dataset_list = [""]

    if cfg.DATA.DATASETS.NAMES in torchvision_dataset_list:
        return make_data_loader_for_classic_datasets(cfg, for_train)

    train_transforms = build_transforms(cfg, is_train=for_train)
    val_transforms = build_transforms(cfg, is_train=False)
    test_transforms = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATA.DATASETS.NAMES, root=cfg.DATA.DATASETS.ROOT_DIR)
    classes_list = dataset.category  # 建立classes_list
    num_classes = dataset.num_categories

    #train set # ？？是否要进行label-smoothing
    train_set = ImageDataset(dataset.train, train_transforms)
    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH,
        sampler=RandomSampler(dataset.train, cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH,
                              cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=for_train),
        num_workers=num_workers, collate_fn=collate_fn
    )

    #val set
    val_set = ImageDataset(dataset.val, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        #CJY at 2019.9.26 为了能够平衡样本
        sampler=RandomSampler(dataset.val, cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH,
                              cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=False),
        collate_fn=collate_fn
    )

    #test_set
    test_set = ImageDataset(dataset.test, test_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        # CJY at 2019.9.26 为了能够平衡样本
        sampler=RandomSampler(dataset.test, cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH,
                              cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, num_classes, is_train=False),
        collate_fn=collate_fn
    )
    #notes:
    #1.collate_fn是自定义函数，对提取的batch做处理，例如分开image和label
    return train_loader, val_loader, test_loader, classes_list


#CJY at 2020.1.8  用于弱监督，引入segmentation_loader
def make_seg_data_loader(cfg, for_train):
    if cfg.DATA.DATASETS.SEG_NAMES == "none" or cfg.TRAIN.DATALOADER.MASK_PER_BATCH == 0:  #如果batch为0，那么就返回空  CJY at 2020.7.12
        classes_list = []
        return None, None, None, classes_list

    train_transforms = build_seg_transforms(cfg, is_train=for_train, type="img")
    val_transforms = build_seg_transforms(cfg, is_train=False, type="img")
    test_transforms = build_seg_transforms(cfg, is_train=False, type="img")
    train_mask_transforms = build_seg_transforms(cfg, is_train=for_train, type="mask")
    val_mask_transforms = build_seg_transforms(cfg, is_train=False, type="mask")
    test_mask_transforms = build_seg_transforms(cfg, is_train=False, type="mask")

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATA.DATASETS.SEG_NAMES, root=cfg.DATA.DATASETS.ROOT_DIR)
    classes_list = dataset.category  # 建立classes_list
    num_classes = dataset.num_categories

    # train set
    # 是否要进行label-smoothing
    train_set = SegmentationDataset(dataset.seg_train, train_transforms, train_mask_transforms, cfg, is_train=for_train)
    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.MASK_PER_BATCH, num_workers=num_workers, #shuffle=True,
        # CJY  为了保证类别均衡
        sampler=RandomSamplerForSegmentation(dataset.seg_train, 1, 1, num_classes, is_train=for_train),    #此处让所有分类标签的按顺序以1为单位交替进行
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
    return train_loader, val_loader, test_loader, classes_list
