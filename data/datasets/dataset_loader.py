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
    def __init__(self, dataset, transform=None, target_transform=None, cfg=None, is_train=False):  # root表示图片路径
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train

        self.ratio = cfg.DATA.TRANSFORM.MASK_SIZE_RATIO
        self.padding = cfg.DATA.TRANSFORM.PADDING   #不引入padding和crop
        if self.is_train == True:
            self.pad_num = self.padding * 2 - 1  #为了后面直接使用 加入 *2-1， 用于生成随机数
        else:
            self.pad_num = 0 #self.padding * 2 - 1  #为了后面直接使用 加入 *2-1， 用于生成随机数
        self.resizeH = cfg.DATA.TRANSFORM.SIZE[0]
        self.resizeW = cfg.DATA.TRANSFORM.SIZE[1]
        self.mask_resizeH = self.resizeH // self.ratio
        self.mask_resizeW = self.resizeW // self.ratio
        self.MaxPool = torch.nn.AdaptiveMaxPool2d((self.mask_resizeH, self.mask_resizeW))
        self.MaskPad = torch.nn.ZeroPad2d(self.padding//self.ratio)   #padding最好是ratio的倍数

        # 用于生成mask的维度
        self.seg_num_classes = cfg.MODEL.SEG_NUM_CLASSES


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
            mask_list.append(mask[0:1])    # cjy 由于读进来的可能是3通道，所以增加[0:1]
        mask = torch.cat(mask_list)


        #上面让mask读入的为原图尺寸标签
        mask = self.MaxPool(mask)

        # 随机剪切（不过像素级标签剪切平移是不是没什么用）  CJY 2020.8.3 目前处于关闭状态，也就是说没有数据扩增
        #"""
        if self.pad_num > 0:
            mask = self.MaskPad(mask)
            randH = random.randint(0, self.pad_num)
            randW = random.randint(0, self.pad_num)
        else:
            randH = 0
            randW = 0
        mask_randH = randH//self.ratio
        mask_randW = randW//self.ratio
        img = img[:, randH:randH + self.resizeH, randW:randW + self.resizeW]
        mask = mask[:, mask_randH:mask_randH + self.mask_resizeH, mask_randW:mask_randW + self.mask_resizeW]
        #"""

        if self.seg_num_classes == 1:
            mask = torch.max(mask, dim=0, keepdim=True)[0]
        elif self.seg_num_classes != mask.shape[0]:
            raise Exception("SEG_NUM_CLASSES cannot match the channels of mask")


        return img, mask, img_label, image_path  # 返回的是图片


