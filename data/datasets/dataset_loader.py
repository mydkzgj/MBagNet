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
            self.pad_num = 0
        self.resizeH = cfg.DATA.TRANSFORM.SIZE[0]
        self.resizeW = cfg.DATA.TRANSFORM.SIZE[1]
        self.mask_resizeH = self.resizeH // self.ratio
        self.mask_resizeW = self.resizeW // self.ratio
        self.MaxPool = torch.nn.AdaptiveMaxPool2d((self.mask_resizeH, self.mask_resizeW))
        self.MaskPad = torch.nn.ZeroPad2d(self.padding//self.ratio)   #padding最好是ratio的倍数

        # 用于生成mask的维度
        self.seg_num_classes = cfg.MODEL.SEG_NUM_CLASSES

        # CJY at 2020.8.12
        self.generateColormask = 1
        if self.generateColormask == True:
            import torchvision.transforms as T
            normalize_transform = T.Normalize(mean=cfg.DATA.TRANSFORM.PIXEL_MEAN, std=cfg.DATA.TRANSFORM.PIXEL_STD)
            self.norm_transform = T.Compose([normalize_transform])

            from ..transforms.transforms import RandomErasing
            if self.is_train == True:
                m_pad = cfg.DATA.TRANSFORM.PADDING * 4
                prob = cfg.TRAIN.TRANSFORM.PROB
                re_prob = cfg.TRAIN.TRANSFORM.RE_PROB
                self.shuffle_th = 0 #0.5
                self.pick_channel_th = -1
            else:
                m_pad = 0
                prob = 0
                re_prob = 0
                self.shuffle_th = 1
                self.pick_channel_th = 1

            self.single_mask_transform = T.Compose([
                # T.Resize(cfg.DATA.TRANSFORM.SIZE),
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=prob),
                T.Pad(m_pad),  # 暂时先去掉padding，因为有可能让mask中的病灶全部被剪切去
                T.RandomCrop(cfg.DATA.TRANSFORM.SIZE),
                T.ToTensor(),
                RandomErasing(probability=re_prob, sl=0.2, sh=0.5, mean=[0])
            ])

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
        mask = self.MaxPool(mask)  #mask的resize选择maxpool

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


        # CJY at 2020.8.12 在dataloader中生成colormask
        if self.generateColormask == True:
            # CJY at 2020.9.2
            # 将每一维得mask进行单独扩增
            mask_copy = mask.clone()
            for i in range(mask.shape[0]):
                mask[i:i+1] = self.single_mask_transform(mask[i:i+1])

            import os
            _, imgName = os.path.split(image_path)
            lesionTypeList = ["MA", "EX", "SE", "HE"]
            ColorMap = {
                "MA": torch.Tensor([0,255,0]).unsqueeze(1).unsqueeze(1)/255.0,
                "EX": torch.Tensor([255,255,0]).unsqueeze(1).unsqueeze(1)/255.0,
                "SE": torch.Tensor([255,165,0]).unsqueeze(1).unsqueeze(1)/255.0,
                "HE": torch.Tensor([255,0,0]).unsqueeze(1).unsqueeze(1)/255.0,
            }

            drawOrder = ["HE", "SE", "EX", "MA"]
            canvas = torch.zeros_like(img)

            # CJY at 2020.9.2  将mask按照draw order进行重叠位置得筛除（保留后来的，使得没有重叠位置）
            mask_no_overlap = mask.clone()
            if "HE" in imgName or "SE" in imgName or "EX" in imgName or "MA" in imgName:
                for lesionType in drawOrder:
                    index = lesionTypeList.index(lesionType)
                    if lesionType in imgName:
                        mask_no_overlap = mask_no_overlap * (1 - mask[index:index + 1])
                        mask_no_overlap[index:index + 1] = mask[index:index + 1]
                    else:
                        mask_no_overlap[index:index + 1] = mask_no_overlap[index:index + 1] * 0
            else:
                for lesionType in drawOrder:
                    index = lesionTypeList.index(lesionType)
                    mask_no_overlap = mask_no_overlap * (1 - mask[index:index + 1])
                    mask_no_overlap[index:index + 1] = mask[index:index + 1]

            # 只挑选其中一个有值通道
            """
            if random.random() > self.pick_channel_th:
                non_zero_channel_index = []
                for index, lesionType in enumerate(lesionTypeList):
                    sum = mask_no_overlap[index:index + 1].sum()
                    if sum != 0:
                        non_zero_channel_index.append(index)

                if non_zero_channel_index != []:  # != 与 is not 有何区别，为什么用后者会和num_worker冲突，报错
                    random.shuffle(non_zero_channel_index)
                    #i = random.randint(0, len(non_zero_channel_index) - 1)
                    pick_channel = non_zero_channel_index[0]

                    for index, lesionType in enumerate(lesionTypeList):
                        if index != pick_channel:
                            mask_no_overlap[index:index + 1] = mask_no_overlap[index:index + 1] * 0
            #"""


            # 将通道随机打乱
            if random.random() > self.shuffle_th:
                mask_no_overlap_list = [mask_no_overlap[i:i + 1] for i in range(mask_no_overlap.shape[0])]
                random.shuffle(mask_no_overlap_list)
                mask_no_overlap = torch.cat(mask_no_overlap_list, dim=0)

            # 依照mask_no_overlap进行绘制
            for index, lesionType in enumerate(lesionTypeList):
                #index = lesionTypeList.index(lesionType)
                sum = mask_no_overlap[index:index + 1].sum()
                if sum == 0 and img_label[index] == 1:
                    img_label[index] = 0   # 当病灶在边缘时可能会出现没有crop到他的问题
                elif sum != 0 and img_label[index] == 0:
                    img_label[index] = 1

                if img_label[index] == 1:
                    canvas = canvas * (1 - mask_no_overlap[index:index + 1]) + mask_no_overlap[index:index + 1] * ColorMap[lesionTypeList[index]]

                sum_mask = torch.sum(mask_no_overlap, dim=0, keepdim=True).ne(0).float()

            img = self.norm_transform(canvas) * sum_mask
            mask = mask_copy

        return img, mask, img_label, image_path  # 返回的是图片




