# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .datasets import init_dataset, ImageDataset
from .samplers import RandomSampler
from .transforms import build_transforms

import torchvision
from torchvision import transforms

import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    imgs, labels, _, = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels

def make_data_loader(cfg):
    # CJY at 2019.11.20 加入其他非医学图像数据集
    if cfg.DATA.DATASETS.NAMES in ["cifa10"]:
        BATCH_SIZE = 128  # 批处理尺寸(batch_size)
        LR = 0.01  # 学习率
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


    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    test_transforms = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    if len(cfg.DATA.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATA.DATALOADER.NAMES, root=cfg.DATA.DATALOADER.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATA.DATASETS.NAMES, root=cfg.DATA.DATASETS.ROOT_DIR)


    num_classes = dataset.num_categories
    # 建立classes_list
    classes_list = dataset.category

    #train set
    #是否要进行label-smoothing
    #train_set = ImageDataset(dataset.train, train_transforms)
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATA.DATALOADER.SAMPLER == 'softmax':
        """
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn_classifaction
        )
        """
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH,
            sampler=RandomSampler(dataset.train, cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH,
                                  cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, dataset.num_categories,
                                  is_train=True),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH,
            sampler=RandomSampler(dataset.train, cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH, cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, dataset.num_categories, is_train=True),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=collate_fn
        )

    #val set
    #val_set = ImageDataset(dataset.val, val_transforms)
    val_set = ImageDataset(dataset.val, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        #CJY at 2019.9.26 为了能够平衡样本
        sampler=RandomSampler(dataset.val, cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH, cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, dataset.num_categories, is_train=False),
        collate_fn=collate_fn
    )

    #test_set
    #test_set = ImageDataset(dataset.test, test_transforms)
    test_set = ImageDataset(dataset.test, test_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        # CJY at 2019.9.26 为了能够平衡样本
        sampler=RandomSampler(dataset.test, cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH, cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, dataset.num_categories, is_train=False),
        collate_fn=collate_fn
    )
    #notes:
    #1.collate_fn是自定义函数，对提取的batch做处理，例如分开image和label

    return train_loader, val_loader, test_loader, classes_list
