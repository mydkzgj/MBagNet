# encoding: utf-8
"""
@author:  chenjiayang
@contact: sychenjiayang@163.com
"""

import torchvision.transforms as T
import data.transforms.cla_transforms as CT
import data.transforms.seg_transforms as ST


#from .cla_transforms import RandomErasing, PaddingToSquare

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = T.Compose([
            CT.PaddingToSquare(padding_mode=cfg.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE),
            T.Resize(cfg.DATA.TRANSFORM.SIZE),
            T.RandomHorizontalFlip(p=cfg.DATA.TRANSFORM.PROB),
            T.Pad(cfg.DATA.TRANSFORM.PADDING),
            T.RandomCrop(cfg.DATA.TRANSFORM.SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.DATA.TRANSFORM.PIXEL_MEAN, std=cfg.DATA.TRANSFORM.PIXEL_STD),
            T.RandomErasing(p=cfg.DATA.TRANSFORM.RE_PROB)  # 由于之前已经做过归一化，所以v设置为0即可
            #RandomErasing(probability=cfg.DATA.TRANSFORM.RE_PROB, mean=cfg.DATA.TRANSFORM.PIXEL_MEAN)  #是不是应该在归一化之前
        ])
    else:
        transform = T.Compose([
            CT.PaddingToSquare(padding_mode=cfg.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE),
            T.Resize(cfg.DATA.TRANSFORM.SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.DATA.TRANSFORM.PIXEL_MEAN, std=cfg.DATA.TRANSFORM.PIXEL_STD)
        ])

    return transform


def build_seg_transforms(cfg, is_train=True):
    #ratio = cfg.DATA.TRANSFORM.MASK_SIZE_RATIO
    #mask_size = (cfg.DATA.TRANSFORM.SIZE[0]//ratio, cfg.DATA.TRANSFORM.SIZE[1]//ratio)
    if is_train:
        transform = ST.Compose([
            ST.PaddingToSquare(padding_mode=cfg.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE),
            ST.Resize(cfg.DATA.TRANSFORM.SIZE),
            ST.RandomHorizontalFlip(p=cfg.DATA.TRANSFORM.PROB),
            ST.Pad(cfg.DATA.TRANSFORM.PADDING),
            ST.RandomCrop(cfg.DATA.TRANSFORM.SIZE),
            ST.ToTensor(),
            ST.Normalize(mean=cfg.DATA.TRANSFORM.PIXEL_MEAN, std=cfg.DATA.TRANSFORM.PIXEL_STD),
            ST.RandomErasing(p=cfg.DATA.TRANSFORM.RE_PROB,)  # 由于之前已经做过归一化，所以v设置为0即可
        ])
    else:
        transform = T.Compose([
            ST.PaddingToSquare(padding_mode=cfg.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE),
            ST.Resize(cfg.DATA.TRANSFORM.SIZE),
            ST.ToTensor(),
            ST.Normalize(mean=cfg.DATA.TRANSFORM.PIXEL_MEAN, std=cfg.DATA.TRANSFORM.PIXEL_STD),
        ])

    return transform


"""
def build_seg_transforms(cfg, is_train=True, type="img"):  #去除随机因素   #img, mask, together
    mean = cfg.DATA.TRANSFORM.PIXEL_MEAN
    std = cfg.DATA.TRANSFORM.PIXEL_STD
    normalize_transform = T.Normalize(mean=mean, std=std)
    ratio = cfg.DATA.TRANSFORM.MASK_SIZE_RATIO
    mask_size = (cfg.DATA.TRANSFORM.SIZE[0]//ratio, cfg.DATA.TRANSFORM.SIZE[1]//ratio)

    if is_train:  #差别就在于for train img加入了Padding
        if type == "img":
            transform = T.Compose([
                T.Resize(cfg.DATA.TRANSFORM.SIZE),
                # T.RandomHorizontalFlip(p=cfg.DATA.TRANSFORM.PROB),
                T.Pad(cfg.DATA.TRANSFORM.PADDING),  #暂时先去掉padding，因为有可能让mask中的病灶全部被剪切去
                # T.RandomCrop(cfg.DATA.TRANSFORM.SIZE),
                T.ToTensor(),
                normalize_transform,
                # RandomErasing(probability=cfg.DATA.TRANSFORM.RE_PROB, mean=cfg.DATA.TRANSFORM.PIXEL_MEAN)
            ])
        elif type == "mask":
            transform = T.Compose([
                # T.Resize(mask_size), #interpolation=Image.ANTIALIAS),#,Image.NEAREST),   #对于掩膜标签 应该不改变标签值，使用最邻近插值
                # T.Pad(cfg.DATA.TRANSFORM.PADDING),
                T.ToTensor(),
            ])
    else:
        if type == "img":
            transform = T.Compose([
                T.Resize(cfg.DATA.TRANSFORM.SIZE),
                # T.RandomHorizontalFlip(p=cfg.DATA.TRANSFORM.PROB),
                # T.Pad(cfg.DATA.TRANSFORM.PADDING),  #暂时先去掉padding，因为有可能让mask中的病灶全部被剪切去
                # T.RandomCrop(cfg.DATA.TRANSFORM.SIZE),
                T.ToTensor(),
                normalize_transform,
                # RandomErasing(probability=cfg.DATA.TRANSFORM.RE_PROB, mean=cfg.DATA.TRANSFORM.PIXEL_MEAN)
            ])
        elif type == "mask":   #mask不做resize和padding，外面做
            transform = T.Compose([
                # T.Resize(mask_size), #interpolation=Image.ANTIALIAS),#,Image.NEAREST),   #对于掩膜标签 应该不改变标签值，使用最邻近插值
                # T.Pad(cfg.DATA.TRANSFORM.PADDING),
                T.ToTensor(),
            ])

    return transform
"""