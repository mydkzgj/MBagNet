# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import random



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Fundus Dataset from TongRen """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, img_label = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_label, img_path



# data.Dataset:

# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)


class SegmentationDataset(Dataset):

    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, dataset, transform=None, target_transform=None, cfg=None):  # root表示图片路径
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

        self.ratio = cfg.DATA.TRANSFORM.MASK_SIZE_RATIO
        self.padding = cfg.DATA.TRANSFORM.PADDING * self.ratio * 2 -1  #为了后面直接使用 加入 *2-1
        self.resizeH = cfg.DATA.TRANSFORM.SIZE[0]
        self.resizeW = cfg.DATA.TRANSFORM.SIZE[1]
        self.mask_resizeH = self.resizeH // self.ratio
        self.mask_resizeW = self.resizeW // self.ratio


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, mask_path_list, img_label = self.dataset[index]
        #print(image_path)
        img = read_image(image_path)
        if self.transform is not None:
            img = self.transform(img)

        mask_list = []
        for mask_path in mask_path_list:
            mask = Image.open(mask_path)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            mask_list.append(mask)
        mask = torch.cat(mask_list)

        randH = random.randint(0, self.padding)
        randW = random.randint(0, self.padding)

        mask_randH = randH//self.ratio
        mask_randW = randW//self.ratio

        img = img[:, randH:randH + self.resizeH, randW:randW + self.resizeW]
        mask = mask[:, mask_randH:mask_randH + self.mask_resizeH, mask_randW:mask_randW + self.mask_resizeW]

        return img, mask, img_label  # 返回的是图片


