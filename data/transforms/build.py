# encoding: utf-8
"""
@author:  chenjiayang
@contact: sychenjiayang@163.com
"""
from PIL import Image
import torchvision.transforms as T
from . import cla_transforms as CT
from . import seg_transforms as ST


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


def build_transforms_for_colormask(cfg, is_train=True):
    #ratio = cfg.DATA.TRANSFORM.MASK_SIZE_RATIO
    #mask_size = (cfg.DATA.TRANSFORM.SIZE[0]//ratio, cfg.DATA.TRANSFORM.SIZE[1]//ratio)
    if is_train:
        transform = ST.Compose([
            T.PaddingToSquare(padding_mode=cfg.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE),
            T.Resize(cfg.DATA.TRANSFORM.SIZE),
            T.RandomHorizontalFlip(p=cfg.DATA.TRANSFORM.PROB),
            T.Pad(cfg.DATA.TRANSFORM.PADDING),
            T.RandomCrop(cfg.DATA.TRANSFORM.SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.DATA.TRANSFORM.PIXEL_MEAN, std=cfg.DATA.TRANSFORM.PIXEL_STD),
            T.RandomErasing(p=cfg.DATA.TRANSFORM.RE_PROB,)  # 由于之前已经做过归一化，所以v设置为0即可
        ])

        target_transform = ST.Compose([
            T.PaddingToSquare(padding_mode=cfg.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE),
            T.Resize(cfg.DATA.TRANSFORM.SIZE, interpolation=Image.BOX),
            T.RandomHorizontalFlip(p=cfg.DATA.TRANSFORM.PROB),
            T.Pad(cfg.DATA.TRANSFORM.PADDING),
            T.RandomCrop(cfg.DATA.TRANSFORM.SIZE),
            T.ToTensor(),
            T.RandomErasing(p=cfg.DATA.TRANSFORM.RE_PROB,)  # 由于之前已经做过归一化，所以v设置为0即可
        ])

    else:
        transform = T.Compose([
            T.PaddingToSquare(padding_mode=cfg.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE),
            T.Resize(cfg.DATA.TRANSFORM.SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.DATA.TRANSFORM.PIXEL_MEAN, std=cfg.DATA.TRANSFORM.PIXEL_STD),
        ])

        target_transform = T.Compose([
            T.PaddingToSquare(padding_mode=cfg.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE),
            T.Resize(cfg.DATA.TRANSFORM.SIZE),
            T.ToTensor(),
        ])

    return transform, target_transform